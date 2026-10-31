"""Network cost model, link-failure simulation, offloading policies and metrics.

All policies are vectorised over sensing windows. Model outputs are pre-computed,
so each policy/failure combination is evaluated in milliseconds.
"""
import itertools
import numpy as np

# One threshold grid shared by every policy, so no method gets a finer operating-point search.
THETA_GRID = np.round(np.linspace(0.05, 0.95, 37), 4)

# ------------------------------------------------------------------ network


def net_costs(cfg, tau_mc_ms, tau_det_ms, C):
    """Latency (ms) and gateway energy (mJ) components for each path (Sec. 3.3)."""
    D = cfg.window * C * cfg.bytes_per_value                # bytes sent per window
    tx_gf = D * 8 / cfg.bw_gf_kbps                          # ms  (bits / kbit/s = ms)
    tx_fc = D * 8 / cfg.bw_fc_kbps
    fog_extra = tx_gf + cfg.rtt_gf_ms + cfg.fog_ms
    cloud_extra = tx_fc + cfg.rtt_fc_ms + cfg.cloud_ms
    return dict(
        tau_mc=tau_mc_ms, tau_det=tau_det_ms, fog_extra=fog_extra, cloud_extra=cloud_extra,
        cloud_only=fog_extra - cfg.fog_ms + cloud_extra,   # relayed via fog, no fog model
        e_cpu_mc=cfg.p_cpu_w * tau_mc_ms, e_cpu_det=cfg.p_cpu_w * tau_det_ms,
        e_tx=cfg.p_tx_w * tx_gf, lmax=cfg.latency_max_ms)


def failure_masks(data, idx, rho, kind, rng):
    """Availability (1 = up) of gateway-fog and fog-cloud links for each window."""
    n = len(idx)
    if rho <= 0:
        return np.ones(n, bool), np.ones(n, bool)
    if kind == "independent":
        p = np.full(n, rho)
    elif kind == "correlated":
        b, t = data["b"][idx], data["t"][idx]
        rain = data["rain"]
        r3 = rain[b, t] + rain[b, np.maximum(t - 1, 0)] + rain[b, np.maximum(t - 2, 0)]
        w = r3 + 1.0
        p = np.clip(rho * w / w.mean(), 0, 0.95)          # mean failure rate ~ rho, higher in storms
    else:
        raise ValueError(kind)
    return rng.random(n) >= p, rng.random(n) >= p


# ------------------------------------------------------------------ policies


def edgewarn(o, a_gf, a_fc, P, net, fallback=True):
    ps, us, pf, uf, pt = o["ps"], o["us"], o["pf"], o["uf"], o["pt"]
    eta = P["eta0"] * (1 - P["beta"] * ps)
    esc = (us > eta) & (ps >= P["theta_low"])
    fog_ok = net["tau_mc"] + net["fog_extra"] <= net["lmax"]
    cloud_ok = net["tau_mc"] + net["fog_extra"] + net["cloud_extra"] <= net["lmax"]
    to_fog = esc & a_gf & fog_ok
    eta_f = P["eta0"] * (1 - P["beta"] * pf)
    to_cloud = to_fog & (uf > eta_f) & a_fc & cloud_ok
    p = ps.copy()
    p[to_fog] = pf[to_fog]
    p[to_cloud] = pt[to_cloud]
    thr = np.full(len(p), P["theta"])
    local = esc & ~to_fog
    if fallback:
        thr[local] = P["theta_safe"]
    warn = p >= thr
    lat = net["tau_mc"] + to_fog * net["fog_extra"] + to_cloud * net["cloud_extra"]
    en = net["e_cpu_mc"] + to_fog * net["e_tx"]
    return dict(warn=warn, lat=lat, en=en, esc=to_fog, has_pred=np.ones(len(p), bool))


def cloud_only(o, a_gf, a_fc, P, net):
    up = a_gf & a_fc
    warn = up & (o["pt"] >= P["theta"])
    n = len(warn)
    return dict(warn=warn, lat=np.full(n, net["cloud_only"]), en=np.full(n, net["e_tx"]),
                esc=np.ones(n, bool), has_pred=up)


def edge_only(o, a_gf, a_fc, P, net, key="pd"):
    warn = o[key] >= P["theta"]
    n = len(warn)
    return dict(warn=warn, lat=np.full(n, net["tau_det"]), en=np.full(n, net["e_cpu_det"]),
                esc=np.zeros(n, bool), has_pred=np.ones(n, bool))


def fixed_conf(o, a_gf, a_fc, P, net):
    """Confidence-gated cascade: escalate to the cloud when p is near the threshold."""
    pd, pt = o["pd"], o["pt"]
    esc = np.abs(pd - P["theta"]) < P["band"]
    go = esc & a_gf & a_fc
    p = np.where(go, pt, pd)
    warn = p >= P["theta"]
    lat = net["tau_det"] + go * net["cloud_only"]          # relayed through fog to the teacher
    en = net["e_cpu_det"] + go * net["e_tx"]
    return dict(warn=warn, lat=lat, en=en, esc=go, has_pred=np.ones(len(p), bool))


# ------------------------------------------------------------------ metrics


def event_index(data, split=2):
    """Flood onsets in a split and, for each, the sample indices allowed to warn for it."""
    B, T, H = data["B"], data["T"], data["H"]
    pos = -np.ones((B, T), int)
    sel = np.where(data["split"] == split)[0]
    pos[data["b"][sel], data["t"][sel]] = sel
    ex = data["exceed"]
    events = []
    for b in range(B):
        onsets = np.where((ex[b, 1:] == 1) & (ex[b, :-1] == 0))[0] + 1
        for te in onsets:
            cand = [pos[b, t] for t in range(te - H, te) if t >= 0 and pos[b, t] >= 0]
            if len(cand) == H:                       # all candidate windows inside this split
                events.append((te, np.array(cand)))
    return events


def metrics(res, y, events, idx_global_to_local, cost=None):
    warn = res["warn"]
    tp = np.sum(warn & (y == 1)); fp = np.sum(warn & (y == 0))
    fn = np.sum(~warn & (y == 1)); tn = np.sum(~warn & (y == 0))
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    far = fp / max(fp + tn, 1)
    det, leads = 0, []
    for te, cand in events:
        loc = idx_global_to_local[cand]                 # ordered t = te-H .. te-1
        w = warn[loc]
        if w.any():
            det += 1
            first = np.argmax(w)                        # earliest warning
            leads.append((len(cand) - first) * 24.0)    # hours ahead of onset (daily data)
    out = dict(precision=prec, recall=rec, f1=f1, far=far,
               availability=100.0 * det / max(len(events), 1),
               lead_h=float(np.mean(leads)) if leads else 0.0,
               latency_ms=float(res["lat"].mean()), energy_mJ=float(res["en"].mean()),
               escalated_pct=100.0 * float(res["esc"].mean()))
    if cost is not None:
        out["cost"] = window_cost(res, y, cost)
    return out


def window_cost(res, y, c):
    miss = (~res["warn"]) & (y == 1)
    fa = res["warn"] & (y == 0)
    return float(np.mean(c["lambda_lat"] * res["lat"] + c["lambda_energy"] * res["en"]
                         + c["cost_miss"] * miss + c["cost_fa"] * fa))


# ------------------------------------------------------------------ tuning


def tune_threshold(fn, o, y, net, cost, extra=None):
    best = None
    ones = np.ones(len(y), bool)
    for th in THETA_GRID:
        P = dict(theta=th, **(extra or {}))
        c = window_cost(fn(o, ones, ones, P, net), y, cost)
        if best is None or c < best[0]:
            best = (c, P)
    return best[1]


def tune_fixed(o, y, net, cost):
    best = None
    ones = np.ones(len(y), bool)
    for th, band in itertools.product(THETA_GRID, [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]):
        P = dict(theta=th, band=band)
        c = window_cost(fixed_conf(o, ones, ones, P, net), y, cost)
        if best is None or c < best[0]:
            best = (c, P)
    return best[1]


def tune_edgewarn(o, y, net, cost, masks_fail, beta_grid=(0.0, 0.5, 0.8)):
    """Stage 1: theta, eta0, beta, theta_low with all links up.
    Stage 2: theta_safe under the tuning failure masks."""
    ones = np.ones(len(y), bool)
    best = None
    for th, e0, be, tl in itertools.product(THETA_GRID,
                                            [0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
                                            beta_grid, [0.01, 0.05, 0.1]):
        P = dict(theta=th, eta0=e0, beta=be, theta_low=tl, theta_safe=th)
        c = window_cost(edgewarn(o, ones, ones, P, net), y, cost)
        if best is None or c < best[0]:
            best = (c, P)
    P = dict(best[1])
    a_gf, a_fc = masks_fail
    best2 = None
    for ts in np.linspace(0.02, P["theta"], 12):
        P["theta_safe"] = ts
        c = window_cost(edgewarn(o, a_gf, a_fc, P, net), y, cost)
        if best2 is None or c < best2[0]:
            best2 = (c, ts)
    P["theta_safe"] = best2[1]
    return P
