"""Data loading, labelling and window construction.

Two sources:
  * synthetic  - a conceptual rainfall-runoff simulator used ONLY to test the
                 pipeline end to end. Do not report results from it.
  * camels     - CAMELS-US daily Daymet forcing + USGS streamflow.

Every source is turned into the same structure:
  forcing   [B, T, 4]  precipitation, mean temperature, solar radiation, vapour pressure
  q         [B, T]     streamflow (NaN where missing)
  coords    [B, 2]     lat, lon (synthetic: positions along river chains)
  basin_ids list of B strings
"""
import os
import glob
import numpy as np

# --------------------------------------------------------------------------
# Synthetic rainfall-runoff generator (pipeline test only)
# --------------------------------------------------------------------------

def synth_dataset(n_basins, n_days, seed=0):
    rng = np.random.default_rng(seed)
    n_chains = max(2, n_basins // 5)
    chain = np.arange(n_basins) % n_chains           # which river a basin is on
    pos = np.arange(n_basins) // n_chains            # position along the river (0 = upstream)
    coords = np.stack([chain * 1.0, pos * 0.5], 1)

    t = np.arange(n_days)
    season = 0.5 + 0.5 * np.sin(2 * np.pi * (t - 120) / 365.25)      # wet season peak
    # Regional storm field per chain, then local noise -> spatially correlated rain
    regional = np.zeros((n_chains, n_days))
    for c in range(n_chains):
        storm = rng.random(n_days) < (0.05 + 0.15 * season)
        regional[c] = storm * rng.gamma(2.0, 12.0, n_days) * (0.6 + season)
    prcp = np.zeros((n_basins, n_days))
    for b in range(n_basins):
        lag = pos[b] % 2                                  # storms move down the chain
        base = np.roll(regional[chain[b]], lag)
        local = (rng.random(n_days) < 0.15) * rng.gamma(1.5, 4.0, n_days)
        prcp[b] = base * rng.uniform(0.7, 1.3) + local

    tmean = 22 + 8 * np.sin(2 * np.pi * (t - 100) / 365.25)[None] + rng.normal(0, 2, (n_basins, n_days))
    srad = 200 + 80 * np.sin(2 * np.pi * (t - 80) / 365.25)[None] - 1.5 * prcp + rng.normal(0, 15, (n_basins, n_days))
    vp = 1200 + 600 * season[None] + 8 * prcp + rng.normal(0, 80, (n_basins, n_days))

    # Two-store conceptual model with upstream routing
    q = np.zeros((n_basins, n_days))
    k_fast = rng.uniform(0.15, 0.35, n_basins)
    k_slow = rng.uniform(0.01, 0.04, n_basins)
    cap = rng.uniform(80, 160, n_basins)
    order = np.argsort(pos)                               # route upstream first
    for b in order:
        s_soil, s_gw = 40.0, 50.0
        up = [u for u in range(n_basins) if chain[u] == chain[b] and pos[u] == pos[b] - 1]
        for d in range(n_days):
            et = max(0.0, 0.08 * (tmean[b, d] - 5)) * (s_soil / cap[b])
            s_soil = max(0.0, s_soil + prcp[b, d] - et)
            excess = max(0.0, s_soil - cap[b])
            s_soil -= excess
            perc = 0.03 * s_soil
            s_soil -= perc
            s_gw += perc
            qf = k_fast[b] * excess + 0.02 * s_soil ** 1.2 / 10
            qs = k_slow[b] * s_gw
            s_gw -= qs
            inflow = sum(0.6 * q[u, d - 1] for u in up) if d > 0 else 0.0
            q[b, d] = qf + qs + inflow
    q = q * rng.lognormal(0, 0.05, q.shape)               # measurement noise

    forcing = np.stack([prcp, tmean, srad, vp], -1).astype(np.float32)
    return dict(forcing=forcing, q=q.astype(np.float32), coords=coords.astype(np.float32),
                basin_ids=[f"syn{b:03d}" for b in range(n_basins)])


# --------------------------------------------------------------------------
# CAMELS-US loader
# --------------------------------------------------------------------------

def _read_forcing(path):
    # Daymet lumped forcing: 3 header lines (lat, elev, area), then a column header row
    arr = np.genfromtxt(path, skip_header=4)
    ymd = arr[:, 0:3].astype(int)
    prcp, srad, tmax, tmin, vp = arr[:, 5], arr[:, 6], arr[:, 8], arr[:, 9], arr[:, 10]
    tmean = (tmax + tmin) / 2
    return ymd, np.stack([prcp, tmean, srad, vp], -1)


def _read_flow(path):
    rows = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            rows.append((int(p[1]), int(p[2]), int(p[3]), float(p[4])))
    a = np.array(rows)
    q = a[:, 3]
    q[q < 0] = np.nan                                     # -999 = missing
    return a[:, :3].astype(int), q


def load_camels(root, basins_file="", max_basins=50):
    topo = glob.glob(os.path.join(root, "**", "camels_topo.txt"), recursive=True)
    topo_path = topo[0] if topo else os.path.join(root, "camels_topo.txt")
    coords_all = {}
    if os.path.exists(topo_path):
        with open(topo_path) as f:
            header = f.readline().strip().split(";")
            i_id, i_lat, i_lon = header.index("gauge_id"), header.index("gauge_lat"), header.index("gauge_lon")
            for line in f:
                p = line.strip().split(";")
                coords_all[p[i_id].zfill(8)] = (float(p[i_lat]), float(p[i_lon]))
    # CAMELS time-series download: basin_metadata/gauge_information.txt (tab separated)
    for gi in glob.glob(os.path.join(root, "**", "*gauge_information.txt"), recursive=True):
        with open(gi) as f:
            f.readline()
            for line in f:
                p = [x.strip() for x in line.split("\t") if x.strip()]
                if len(p) >= 5:
                    coords_all.setdefault(p[1].zfill(8), (float(p[3]), float(p[4])))

    if basins_file:
        with open(basins_file) as f:
            ids = [l.strip().zfill(8) for l in f if l.strip()]
    else:
        ids = sorted(os.path.basename(p)[:8] for p in
                     glob.glob(os.path.join(root, "usgs_streamflow", "**", "*_streamflow_qc.txt"), recursive=True))
    ids = ids[:max_basins]

    # Read every basin first, then align all of them on the dates they share, so that
    # time index t means the same calendar day in every basin (needed for neighbour
    # features and for a common chronological split).
    raw = []
    for gid in ids:
        fp = glob.glob(os.path.join(root, "basin_mean_forcing", "daymet", "**", f"{gid}_lump_cida_forcing_leap.txt"),
                       recursive=True)
        qp = glob.glob(os.path.join(root, "usgs_streamflow", "**", f"{gid}_streamflow_qc.txt"), recursive=True)
        if not fp or not qp:
            continue
        ymd_f, frc = _read_forcing(fp[0])
        ymd_q, q = _read_flow(qp[0])
        f_map = {tuple(int(v) for v in r): frc[i] for i, r in enumerate(ymd_f)}
        q_map = {tuple(int(v) for v in r): q[i] for i, r in enumerate(ymd_q)}
        raw.append((gid, f_map, q_map))
    if not raw:
        raise FileNotFoundError(f"no CAMELS basins found under {root}")
    common = set(raw[0][1]) & set(raw[0][2])
    for _, f_map, q_map in raw[1:]:
        common &= set(f_map) & set(q_map)
    dates = sorted(common)
    if not dates:
        raise ValueError("the selected basins share no common dates")
    forcing = np.stack([np.stack([f_map[d] for d in dates]) for _, f_map, _ in raw]).astype(np.float32)
    q = np.stack([np.array([q_map[d] for d in dates]) for _, _, q_map in raw]).astype(np.float32)
    kept = [g for g, _, _ in raw]
    coords = [coords_all.get(g, (0.0, 0.0)) for g in kept]
    return dict(forcing=forcing, q=q, coords=np.array(coords, np.float32), basin_ids=kept,
                dates=np.array(dates, dtype=np.int32))



# --------------------------------------------------------------------------
# Feature matrix, labels, windows
# --------------------------------------------------------------------------

def nearest_neighbours(coords, k):
    d = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    k = min(k, len(coords) - 1)
    return np.argsort(d, 1)[:, :k]


def build(ds, cfg):
    forcing, q = ds["forcing"], ds["q"]
    B, T, _ = forcing.shape
    W, H = cfg.window, cfg.horizon
    t_train_end = int(T * cfg.train_frac)
    t_val_end = int(T * (cfg.train_frac + cfg.val_frac))

    # Flood threshold from the training period only
    thr = np.nanpercentile(q[:, :t_train_end], cfg.exceed_pct, axis=1)
    exceed = (q > thr[:, None]).astype(np.int8)
    exceed[np.isnan(q)] = 0

    # Channels: log1p(P), T, SRAD, VP, log1p(Q observed up to t); per-basin z-score on training period
    q_filled = np.where(np.isnan(q), np.nanmedian(q[:, :t_train_end], 1, keepdims=True), q)
    feats = np.concatenate([np.log1p(np.clip(forcing[..., :1], 0, None)), forcing[..., 1:],
                            np.log1p(np.clip(q_filled, 0, None))[..., None]], -1)
    mu = feats[:, :t_train_end].mean(1, keepdims=True)
    sd = feats[:, :t_train_end].std(1, keepdims=True) + 1e-6
    F = ((feats - mu) / sd).astype(np.float32)

    nbr = nearest_neighbours(ds["coords"], cfg.n_neighbours)

    # Window ends t: prediction at t for exceedance in (t, t+H]
    b_idx, t_idx, y, split = [], [], [], []
    for b in range(B):
        for t in range(max(W, getattr(cfg, 'teacher_window', W)) - 1, T - H):
            fut = q[b, t + 1:t + H + 1]
            if np.isnan(fut).all():
                continue
            b_idx.append(b)
            t_idx.append(t)
            y.append(int(exceed[b, t + 1:t + H + 1].max()))
            split.append(0 if t + H < t_train_end else (1 if t + H < t_val_end else 2))
            # windows straddling a boundary are assigned to the later period's
            # side only when their label horizon lies fully inside it
    b_idx, t_idx = np.array(b_idx), np.array(t_idx)
    y, split = np.array(y, np.float32), np.array(split)
    # drop windows whose input overlaps the previous split (prevents leakage)
    starts = t_idx - W + 1
    leak = ((split == 1) & (starts < t_train_end)) | ((split == 2) & (starts < t_val_end))
    keep = ~leak
    rain = forcing[..., 0]
    return dict(F=F, nbr=nbr, b=b_idx[keep], t=t_idx[keep], y=y[keep], split=split[keep],
                exceed=exceed, rain=rain, thr=thr, W=W, H=H, B=B, T=T,
                t_train_end=t_train_end, t_val_end=t_val_end, basin_ids=ds["basin_ids"])


def gather(data, idx, with_neighbours):
    """Windows for sample indices idx.

    with_neighbours = False -> own window          [n, W, C]
                      True  -> own + K neighbours   [n, 1+K, W, C]
                      int L -> (own long window [n, L, C], neighbours [n, K, W, C])
    """
    F, W = data["F"], data["W"]
    b, t = data["b"][idx], data["t"][idx]
    if not isinstance(with_neighbours, bool):
        L = int(with_neighbours)
        long_t = t[:, None] + np.arange(-L + 1, 1)[None, :]
        short_t = t[:, None] + np.arange(-W + 1, 1)[None, :]
        nb = data["nbr"][b]
        return F[b[:, None], long_t], F[nb[:, :, None], short_t[:, None, :]]
    time = t[:, None] + np.arange(-W + 1, 1)[None, :]              # [n, W]
    if with_neighbours:
        nodes = np.concatenate([b[:, None], data["nbr"][b]], 1)     # [n, 1+K]
        return F[nodes[:, :, None], time[:, None, :]]               # [n, 1+K, W, C]
    return F[b[:, None], time]                                      # [n, W, C]


def load(cfg):
    if cfg.data_source == "synthetic":
        ds = synth_dataset(cfg.n_synth_basins, cfg.n_synth_days, cfg.data_seed)
    elif cfg.data_source == "camels":
        ds = load_camels(cfg.camels_root, cfg.basins_file, cfg.max_basins)
    else:
        raise ValueError(cfg.data_source)
    return build(ds, cfg)
