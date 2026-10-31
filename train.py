"""Training: weighted BCE for teacher/fog/baselines, knowledge distillation for the student."""
import numpy as np
import torch
import torch.nn.functional as F

from .data import gather
from .models import set_mc


def _tt(a):
    return tuple(torch.from_numpy(x) for x in a) if isinstance(a, tuple) else torch.from_numpy(a)


def _batches(idx, bs, rng):
    idx = idx.copy()
    rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


def _pos_weight(data, idx):
    y = data["y"][idx]
    pos = max(y.sum(), 1.0)
    return float((len(y) - pos) / pos)


def train_model(model, data, cfg, with_neighbours, seed, epochs, teacher=None, log=None, teacher_input=True, lr=None, samples=None):
    """Train `model`. If `teacher` is given, use the distillation loss of Eq. (L_S)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr = np.where(data["split"] == 0)[0]
    va = np.where(data["split"] == 1)[0]
    if len(va) > 30000:                                   # fixed validation subset for early stopping
        va = np.random.default_rng(0).choice(va, 30000, replace=False)
    pw = torch.tensor(_pos_weight(data, tr))
    opt = torch.optim.Adam(model.parameters(), lr=lr or cfg.lr, weight_decay=1e-5)
    best, best_state = np.inf, None
    for ep in range(epochs):
        model.train()
        spe = samples or cfg.samples_per_epoch
        ep_idx = tr if not spe or len(tr) <= spe else rng.choice(tr, spe, replace=False)
        for bi in _batches(ep_idx, cfg.batch, rng):
            x = _tt(gather(data, bi, with_neighbours))
            y = torch.from_numpy(data["y"][bi])
            z = model(x)
            loss = F.binary_cross_entropy_with_logits(z, y, pos_weight=pw)
            if teacher is not None:
                with torch.no_grad():
                    zt = teacher(_tt(gather(data, bi, teacher_input)))
                T = cfg.kd_T
                pt, ps = torch.sigmoid(zt / T), torch.sigmoid(z / T)
                kl = pt * torch.log((pt + 1e-7) / (ps + 1e-7)) + (1 - pt) * torch.log((1 - pt + 1e-7) / (1 - ps + 1e-7))
                loss = cfg.kd_alpha * loss + (1 - cfg.kd_alpha) * T * T * kl.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        vl = eval_loss(model, data, va, with_neighbours, pw, cfg.batch)
        if log:
            log(f"    epoch {ep + 1}/{epochs}  val loss {vl:.4f}")
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def eval_loss(model, data, idx, with_neighbours, pw, bs):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(idx), bs * 4):
        bi = idx[i:i + bs * 4]
        z = model(_tt(gather(data, bi, with_neighbours)))
        y = torch.from_numpy(data["y"][bi])
        tot += F.binary_cross_entropy_with_logits(z, y, pos_weight=pw, reduction="sum").item()
        n += len(bi)
    return tot / n


@torch.no_grad()
def predict(model, data, idx, with_neighbours, mc_samples=1, bs=1024):
    """Return (mean probability, std across MC passes) as numpy arrays."""
    model.eval()
    set_mc(model, mc_samples > 1)
    means, stds = [], []
    for i in range(0, len(idx), bs):
        x = _tt(gather(data, idx[i:i + bs], with_neighbours))
        ps = torch.stack([torch.sigmoid(model(x)) for _ in range(max(1, mc_samples))])
        means.append(ps.mean(0).numpy())
        stds.append(ps.std(0, unbiased=False).numpy())
    set_mc(model, False)
    return np.concatenate(means), np.concatenate(stds)


def fit_temperature(p, y):
    """Temperature scaling on probabilities (Guo et al. 2017). Returns T."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    best_T, best_nll = 1.0, np.inf
    for T in np.linspace(0.3, 5.0, 95):
        q = 1 / (1 + np.exp(-z / T))
        nll = -np.mean(y * np.log(q + 1e-7) + (1 - y) * np.log(1 - q + 1e-7))
        if nll < best_nll:
            best_T, best_nll = T, nll
    return best_T


def apply_temperature(p, T):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    return 1 / (1 + np.exp(-z / T))
