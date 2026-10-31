"""Teacher, fog and student networks, plus LSTM/GRU edge baselines."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MCDropout(nn.Module):
    """Dropout written as an explicit random mask so it survives ONNX export.

    Active when the module is training OR when `mc` is True (Monte Carlo inference).
    """

    def __init__(self, p):
        super().__init__()
        self.p = p
        self.mc = False

    def forward(self, x):
        if not (self.training or self.mc) or self.p == 0:
            return x
        mask = (torch.rand_like(x) >= self.p).to(x.dtype)
        return x * mask / (1 - self.p)


class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k, dilation, p):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, k, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, k, dilation=dilation)
        self.drop = MCDropout(p)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):                       # x [n, C, W]
        h = F.relu(self.conv1(F.pad(x, (self.pad, 0))))
        h = self.drop(h)
        h = F.relu(self.conv2(F.pad(h, (self.pad, 0))))
        h = self.drop(h)
        return F.relu(h + self.skip(x))


class TCNEncoder(nn.Module):
    def __init__(self, c_in, ch, p, levels=4, k=3):
        super().__init__()
        layers, c = [], c_in
        for i in range(levels):
            layers.append(TCNBlock(c, ch, k, 2 ** i, p))
            c = ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):                       # x [n, W, C]
        return self.net(x.transpose(1, 2))[:, :, -1]        # last time step [n, ch]


class Student(nn.Module):
    """Edge model: sees only its own gateway's window."""

    def __init__(self, c_in, ch, p):
        super().__init__()
        self.enc = TCNEncoder(c_in, ch, p, levels=4)
        self.head = nn.Linear(ch, 1)

    def forward(self, x):                       # [n, W, C] -> logits [n]
        return self.head(self.enc(x)).squeeze(-1)


class GraphTCN(nn.Module):
    """Teacher / fog model: shared TCN encoder per node + attention over neighbours."""

    def __init__(self, c_in, ch, p):
        super().__init__()
        self.enc = TCNEncoder(c_in, ch, p, levels=4)
        self.att = nn.Linear(2 * ch, 1)
        self.head = nn.Sequential(nn.Linear(2 * ch, ch), nn.ReLU(), MCDropout(p), nn.Linear(ch, 1))

    def forward(self, x):                       # [n, 1+K, W, C]
        n, m, W, C = x.shape
        h = self.enc(x.reshape(n * m, W, C)).reshape(n, m, -1)
        h_self, h_nb = h[:, 0], h[:, 1:]
        if h_nb.shape[1] == 0:
            agg = torch.zeros_like(h_self)
        else:
            s = self.att(torch.cat([h_self[:, None].expand_as(h_nb), h_nb], -1)).squeeze(-1)
            agg = (torch.softmax(s, 1)[..., None] * h_nb).sum(1)
        return self.head(torch.cat([h_self, agg], -1)).squeeze(-1)


class LongGraphLSTM(nn.Module):
    """Cloud/fog model: LSTM over a long own-basin window (L days) + attention over
    short neighbour windows encoded by a TCN."""

    def __init__(self, c_in, hidden, ch_nbr, p):
        super().__init__()
        self.lstm = nn.LSTM(c_in, hidden, batch_first=True)
        self.drop = MCDropout(p)
        self.nbr_enc = TCNEncoder(c_in, ch_nbr, p, levels=4)
        self.q = nn.Linear(hidden, ch_nbr)
        self.head = nn.Sequential(nn.Linear(hidden + ch_nbr, hidden), nn.ReLU(), MCDropout(p), nn.Linear(hidden, 1))

    def forward(self, x):
        x_self, x_nb = x                        # [n, L, C], [n, K, W, C]
        out, _ = self.lstm(x_self)
        h = self.drop(out[:, -1])
        n, k, W, C = x_nb.shape
        if k == 0:
            agg = torch.zeros(n, self.q.out_features, device=h.device)
        else:
            hn = self.nbr_enc(x_nb.reshape(n * k, W, C)).reshape(n, k, -1)
            a = torch.softmax((hn * self.q(h)[:, None]).sum(-1) / hn.shape[-1] ** 0.5, 1)
            agg = (a[..., None] * hn).sum(1)
        return self.head(torch.cat([h, agg], -1)).squeeze(-1)


class RNNEdge(nn.Module):
    """LSTM or GRU edge baseline (no distillation)."""

    def __init__(self, c_in, hidden, kind="lstm"):
        super().__init__()
        cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = cls(c_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


def set_mc(model, on):
    for m in model.modules():
        if isinstance(m, MCDropout):
            m.mc = on


def n_params(model):
    return sum(p.numel() for p in model.parameters())
