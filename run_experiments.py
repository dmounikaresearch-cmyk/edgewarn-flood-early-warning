"""Run every experiment in the EdgeWarn manuscript (Tables 2-5, Figs. 2-3).

Examples
  python run_experiments.py --quick                         # pipeline check on synthetic data
  python run_experiments.py --source camels --camels_root /path/to/basin_dataset_public_v1p2
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from edgewarn.config import Config
from edgewarn import data as D
from edgewarn.models import Student, GraphTCN, LongGraphLSTM, RNNEdge, n_params
from edgewarn.train import train_model, predict, fit_temperature, apply_temperature
from edgewarn import quant as Q
from edgewarn import policy as PL


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def calibrate(p_val, y_val, p_test):
    T = fit_temperature(p_val, y_val)
    return apply_temperature(p_val, T), apply_temperature(p_test, T)


def edge_outputs(student, tag, data, cfg, va, te, tr, wdir, C):
    """Export a student to ONNX (MC + deterministic), quantize, return outputs and timings."""
    out = {}
    for mode in ("mc", "det"):
        fp = os.path.join(wdir, f"{tag}_{mode}_fp32.onnx")
        if mode == "mc":
            Q.export_onnx(student, fp, cfg.window, C)
        else:
            student.eval()
            torch.onnx.export(student, torch.zeros(4, cfg.window, C), fp, input_names=["x"],
                              output_names=["logit"], dynamic_axes={"x": {0: "n"}, "logit": {0: "n"}},
                              opset_version=17, dynamo=False)
        i8 = os.path.join(wdir, f"{tag}_{mode}_int8.onnx")
        Q.quantize_int8(fp, i8, data, tr)
        out[f"{mode}_fp32_path"], out[f"{mode}_int8_path"] = fp, i8
    s_mc8, s_det8 = Q.session(out["mc_int8_path"], 2), Q.session(out["det_int8_path"], 2)
    s_mc32, s_det32 = Q.session(out["mc_fp32_path"], 2), Q.session(out["det_fp32_path"], 2)
    S = cfg.mc_samples
    for split, idx in (("val", va), ("test", te)):
        out[f"ps_{split}"], out[f"us_{split}"] = Q.ort_predict_mc(s_mc8, data, idx, S)
        out[f"ps32_{split}"], out[f"us32_{split}"] = Q.ort_predict_mc(s_mc32, data, idx, S)
        out[f"pd_{split}"], _ = Q.ort_predict_mc(s_det8, data, idx, 1)
        out[f"pd32_{split}"], _ = Q.ort_predict_mc(s_det32, data, idx, 1)
    one = {k: Q.session(out[f"{k}_path"], 1) for k in ("mc_int8", "det_int8", "mc_fp32", "det_fp32")}
    out["tau_mc8"] = Q.benchmark_ms(one["mc_int8"], cfg.window, C, S)
    out["tau_det8"] = Q.benchmark_ms(one["det_int8"], cfg.window, C, 1)
    out["tau_mc32"] = Q.benchmark_ms(one["mc_fp32"], cfg.window, C, S)
    out["tau_det32"] = Q.benchmark_ms(one["det_fp32"], cfg.window, C, 1)
    out["sess_mc8_1t"] = one["mc_int8"]
    out["size_int8_kb"] = Q.size_kb(out["det_int8_path"])
    out["size_fp32_kb"] = Q.size_kb(out["det_fp32_path"])
    out["sess_mc8"] = s_mc8
    return out


def assemble(eo, fog, teach, y_val, prefix_ps="ps", prefix_us="us", prefix_pd="pd"):
    """Calibrated output dictionaries for val and test."""
    o = {}
    for key, src in ((prefix_ps, "ps"), (prefix_pd, "pd")):
        v, t = calibrate(eo[f"{key}_val"], y_val, eo[f"{key}_test"])
        o.setdefault("val", {})[src] = v
        o.setdefault("test", {})[src] = t
    o["val"]["us"], o["test"]["us"] = eo[f"{prefix_us}_val"], eo[f"{prefix_us}_test"]
    for d in ("val", "test"):
        o[d].update(pf=fog[d][0], uf=fog[d][1], pt=teach[d])
    return o


def per_basin_f1(warn, y, basins):
    out = {}
    for b in np.unique(basins):
        m = basins == b
        tp = np.sum(warn[m] & (y[m] == 1)); fp = np.sum(warn[m] & (y[m] == 0)); fn = np.sum(~warn[m] & (y[m] == 1))
        out[b] = 2 * tp / max(2 * tp + fp + fn, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "camels"])
    ap.add_argument("--camels_root", default="")
    ap.add_argument("--basins_file", default="")
    ap.add_argument("--max_basins", type=int, default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, nargs="*")
    args = ap.parse_args()

    cfg = Config()
    if args.quick:
        cfg.apply_quick()
    cfg.data_source, cfg.camels_root, cfg.basins_file = args.source, args.camels_root, args.basins_file
    if args.max_basins:
        cfg.max_basins = args.max_basins
    if args.seeds:
        cfg.seeds = tuple(args.seeds)
    cfg.out_dir = args.out
    wdir = os.path.join(cfg.out_dir, "models")
    os.makedirs(wdir, exist_ok=True)
    cfg.to_json(os.path.join(cfg.out_dir, "config.json"))
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    log(f"Loading data ({cfg.data_source})")
    data = D.load(cfg)
    C = data["F"].shape[-1]
    tr, va, te = (np.where(data["split"] == s)[0] for s in (0, 1, 2))
    y_val, y_te = data["y"][va], data["y"][te]
    log(f"  {data['B']} basins, {data['T']} days, windows train/val/test = {len(tr)}/{len(va)}/{len(te)}, "
        f"positive rate test = {y_te.mean():.3f}")
    events = PL.event_index(data, 2)
    g2l = -np.ones(len(data["y"]), int)
    g2l[te] = np.arange(len(te))
    log(f"  {len(events)} flood onsets in the test period")
    cost = dict(lambda_lat=cfg.lambda_lat, lambda_energy=cfg.lambda_energy,
                cost_miss=cfg.cost_miss, cost_fa=cfg.cost_fa)

    rows2, rows3, rows4, rows_mc, rows_f2, rows_f3 = [], [], [], [], [], []
    rows_auc = []
    basin_f1 = defaultdict(list)
    sizes = {}

    import pickle
    for seed in cfg.seeds:
        ck = os.path.join(cfg.out_dir, f"rows_seed{seed}.pkl")
        if os.path.exists(ck):
            log(f"=== seed {seed}: loading saved results ===")
            R = pickle.load(open(ck, "rb"))
            rows_auc.extend(R.get("auc", []))
            for lst, key in ((rows2, "r2"), (rows3, "r3"), (rows4, "r4"), (rows_mc, "mc"), (rows_f2, "f2"), (rows_f3, "f3")):
                lst.extend(R[key])
            for k, v in R["sizes"].items():
                sizes.setdefault(k, []).extend(v)
            for k, v in R["basin_f1"].items():
                basin_f1[k].extend(v)
            continue
        n0 = dict(auc=len(rows_auc), r2=len(rows2), r3=len(rows3), r4=len(rows4), mc=len(rows_mc), f2=len(rows_f2), f3=len(rows_f3))
        sizes_before = {k: len(v) for k, v in sizes.items()}
        bf_before = {k: len(v) for k, v in basin_f1.items()}
        log(f"=== seed {seed} ===")
        torch.manual_seed(seed)
        log("  training teacher")
        TIN, FIN = cfg.teacher_window, cfg.fog_window
        teacher = train_model(LongGraphLSTM(C, cfg.teacher_hidden, 32, cfg.dropout), data, cfg, TIN, seed, cfg.epochs_cloud, log=log, lr=cfg.teacher_lr,
                                samples=cfg.teacher_samples_per_epoch)
        log("  training fog model")
        fog = train_model(LongGraphLSTM(C, cfg.fog_hidden, 16, cfg.dropout), data, cfg, FIN, seed + 100, cfg.epochs_teacher, log=log, lr=cfg.teacher_lr)
        log("  distilling student")
        student = train_model(Student(C, cfg.student_ch, cfg.dropout), data, cfg, False, seed + 200,
                              cfg.epochs_student, teacher=teacher, log=log, teacher_input=TIN)
        log("  training student without distillation")
        student_nokd = train_model(Student(C, cfg.student_ch, cfg.dropout), data, cfg, False, seed + 300,
                                   cfg.epochs_student, log=log)
        log("  training LSTM / GRU edge baselines")
        lstm = train_model(RNNEdge(C, cfg.rnn_hidden, "lstm"), data, cfg, False, seed + 400, cfg.epochs_student)
        gru = train_model(RNNEdge(C, cfg.rnn_hidden, "gru"), data, cfg, False, seed + 500, cfg.epochs_student)

        log("  exporting, quantizing and timing edge models")
        eo = edge_outputs(student, f"student_s{seed}", data, cfg, va, te, tr, wdir, C)
        eo_nokd = edge_outputs(student_nokd, f"nokd_s{seed}", data, cfg, va, te, tr, wdir, C)

        log("  fog / teacher predictions")
        fogp = {d: predict(fog, data, idx, FIN, cfg.mc_samples) for d, idx in (("val", va), ("test", te))}
        pf_val, pf_te = calibrate(fogp["val"][0], y_val, fogp["test"][0])
        fogp = {"val": (pf_val, fogp["val"][1]), "test": (pf_te, fogp["test"][1])}
        tp_val, tp_te = calibrate(predict(teacher, data, va, TIN)[0], y_val, predict(teacher, data, te, TIN)[0])
        teach = {"val": tp_val, "test": tp_te}

        O = assemble(eo, fogp, teach, y_val)
        O32 = assemble(eo, fogp, teach, y_val, "ps32", "us32", "pd32")
        Onokd = assemble(eo_nokd, fogp, teach, y_val)
        rnn = {}
        for name, m in (("lstm", lstm), ("gru", gru)):
            v, t = calibrate(predict(m, data, va, False)[0], y_val, predict(m, data, te, False)[0])
            rnn[name] = {"val": {"pd": v}, "test": {"pd": t}}

        net = PL.net_costs(cfg, eo["tau_mc8"], eo["tau_det8"], C)
        net32 = PL.net_costs(cfg, eo["tau_mc32"], eo["tau_det32"], C)

        log("  tuning thresholds on validation")
        rng = np.random.default_rng(seed + 1000)
        tune_masks = PL.failure_masks(data, va, cfg.tune_rho, "correlated", rng)
        P_cloud = PL.tune_threshold(PL.cloud_only, O["val"], y_val, net, cost)
        P_edge = PL.tune_threshold(PL.edge_only, O["val"], y_val, net, cost)
        P_fixed = PL.tune_fixed(O["val"], y_val, net, cost)
        P_ew = PL.tune_edgewarn(O["val"], y_val, net, cost, tune_masks)
        P_ew_nokd = PL.tune_edgewarn(Onokd["val"], y_val, net, cost, tune_masks)
        P_ew_32 = PL.tune_edgewarn(O32["val"], y_val, net32, cost, tune_masks)
        P_ew_b0 = PL.tune_edgewarn(O["val"], y_val, net, cost, tune_masks, beta_grid=(0.0,))
        P_rnn = {k: PL.tune_threshold(PL.edge_only, rnn[k]["val"], y_val, net, cost) for k in rnn}
        P_nokd_edge = PL.tune_threshold(PL.edge_only, Onokd["val"], y_val, net, cost)
        with open(os.path.join(cfg.out_dir, f"thresholds_seed{seed}.json"), "w") as f:
            json.dump(dict(cloud=P_cloud, edge=P_edge, fixed=P_fixed, edgewarn=P_ew), f, indent=2,
                      default=float)

        T = O["test"]
        up = np.ones(len(te), bool)

        def run(method, a_gf, a_fc):
            if method == "Cloud-only (teacher)":
                return PL.cloud_only(T, a_gf, a_fc, P_cloud, net)
            if method == "Edge-only (student)":
                return PL.edge_only(T, a_gf, a_fc, P_edge, net)
            if method == "Edge LSTM":
                return PL.edge_only(rnn["lstm"]["test"], a_gf, a_fc, P_rnn["lstm"], net)
            if method == "Edge GRU":
                return PL.edge_only(rnn["gru"]["test"], a_gf, a_fc, P_rnn["gru"], net)
            if method == "Edge TCN (no distillation)":
                return PL.edge_only(Onokd["test"], a_gf, a_fc, P_nokd_edge, net)
            if method == "Fixed-confidence offloading":
                return PL.fixed_conf(T, a_gf, a_fc, P_fixed, net)
            if method == "EdgeWarn":
                return PL.edgewarn(T, a_gf, a_fc, P_ew, net)
            raise ValueError(method)

        # threshold-free ranking quality of each model's probability, plus saved predictions
        from sklearn.metrics import average_precision_score, roc_auc_score
        probs = {"Cloud-only (teacher)": T["pt"], "Fog model": T["pf"], "Edge-only (student)": T["pd"],
                 "EdgeWarn": T["ps"], "Edge TCN (no distillation)": Onokd["test"]["pd"],
                 "Edge LSTM": rnn["lstm"]["test"]["pd"], "Edge GRU": rnn["gru"]["test"]["pd"]}
        auc = {k: (average_precision_score(y_te, v), roc_auc_score(y_te, v)) for k, v in probs.items()}
        auc["Fixed-confidence offloading"] = auc["Edge-only (student)"]
        rows_auc.extend(dict(seed=seed, method=k, pr_auc=a, roc_auc=b) for k, (a, b) in auc.items())
        np.savez_compressed(os.path.join(cfg.out_dir, f"predictions_seed{seed}.npz"), y_test=y_te, y_val=y_val,
                            basin_test=data["b"][te], t_test=data["t"][te],
                            **{f"test_{k}": v for k, v in probs.items()},
                            **{f"val_{k}": v for k, v in {"Cloud-only (teacher)": O["val"]["pt"], "Fog model": O["val"]["pf"],
                               "Edge-only (student)": O["val"]["pd"], "EdgeWarn": O["val"]["ps"],
                               "Edge TCN (no distillation)": Onokd["val"]["pd"], "Edge LSTM": rnn["lstm"]["val"]["pd"],
                               "Edge GRU": rnn["gru"]["val"]["pd"]}.items()})

        methods2 = ["Cloud-only (teacher)", "Edge-only (student)", "Edge LSTM", "Edge GRU",
                    "Edge TCN (no distillation)", "Fixed-confidence offloading", "EdgeWarn"]
        for m in methods2:
            r = run(m, up, up)
            rows2.append(dict(seed=seed, method=m, **PL.metrics(r, y_te, events, g2l)))
            for b, f in per_basin_f1(r["warn"], y_te, data["b"][te]).items():
                basin_f1[(m, b)].append(f)
            if m in ("Cloud-only (teacher)", "Edge-only (student)", "Fixed-confidence offloading", "EdgeWarn"):
                mt = PL.metrics(r, y_te, events, g2l)
                rows3.append(dict(seed=seed, method=m, latency_ms=mt["latency_ms"], energy_mJ=mt["energy_mJ"],
                                  escalated_pct=mt["escalated_pct"]))

        sizes.setdefault("Cloud-only (teacher)", []).append(4 * n_params(teacher) / 1024)
        sizes.setdefault("Edge-only (student)", []).append(eo["size_int8_kb"])
        sizes.setdefault("Fixed-confidence offloading", []).append(eo["size_int8_kb"])
        sizes.setdefault("EdgeWarn", []).append(eo["size_int8_kb"])
        sizes.setdefault("_student_fp32_kb", []).append(eo["size_fp32_kb"])
        sizes.setdefault("_edge_ms_det", []).append(eo["tau_det8"])
        sizes.setdefault("_edge_ms_mc", []).append(eo["tau_mc8"])
        sizes.setdefault("_params_teacher", []).append(n_params(teacher))
        sizes.setdefault("_params_fog", []).append(n_params(fog))
        sizes.setdefault("_params_student", []).append(n_params(student))

        log("  ablations and failure sweeps")
        for rep in range(3):
            rng = np.random.default_rng(seed * 10 + rep)
            a_gf, a_fc = PL.failure_masks(data, te, cfg.ablation_rho, "correlated", rng)
            abl = {
                "EdgeWarn (full)": PL.edgewarn(T, a_gf, a_fc, P_ew, net),
                "Without distillation": PL.edgewarn(Onokd["test"], a_gf, a_fc, P_ew_nokd, net),
                "Without quantization": PL.edgewarn(O32["test"], a_gf, a_fc, P_ew_32, net32),
                "Without hazard-dependent threshold (beta = 0)": PL.edgewarn(T, a_gf, a_fc, P_ew_b0, net),
                "Without link-aware fallback": PL.edgewarn(T, a_gf, a_fc, P_ew, net, fallback=False),
            }
            for k, r in abl.items():
                rows4.append(dict(seed=seed, rep=rep, variant=k, **PL.metrics(r, y_te, events, g2l)))

            for kind in ("independent", "correlated"):
                for rho in cfg.failure_rates:
                    rng2 = np.random.default_rng(seed * 1000 + rep * 100 + int(rho * 100))
                    a_gf, a_fc = PL.failure_masks(data, te, rho, kind, rng2)
                    for m in ("Cloud-only (teacher)", "Edge-only (student)", "Fixed-confidence offloading", "EdgeWarn"):
                        mt = PL.metrics(run(m, a_gf, a_fc), y_te, events, g2l)
                        rows_f3.append(dict(seed=seed, rep=rep, kind=kind, rho=rho, method=m,
                                            availability=mt["availability"], f1=mt["f1"], far=mt["far"]))

        # Fig. 2: accuracy-cost trade-off
        for e0 in [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]:
            P = dict(P_ew, eta0=e0)
            mt = PL.metrics(PL.edgewarn(T, up, up, P, net), y_te, events, g2l)
            rows_f2.append(dict(seed=seed, method="EdgeWarn", knob=e0, f1=mt["f1"], escalated_pct=mt["escalated_pct"]))
        for band in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
            P = dict(P_fixed, band=band)
            mt = PL.metrics(PL.fixed_conf(T, up, up, P, net), y_te, events, g2l)
            rows_f2.append(dict(seed=seed, method="Fixed-confidence offloading", knob=band, f1=mt["f1"],
                                escalated_pct=mt["escalated_pct"]))

        # Table 5: Monte Carlo samples
        S_list = (1, 3, 5) if cfg.quick else ((1, 5, 10, 20) if seed == cfg.seeds[0] else ())
        for S in S_list:
            sv = Q.ort_predict_mc(eo["sess_mc8"], data, va, S)
            st = Q.ort_predict_mc(eo["sess_mc8"], data, te, S)
            pv, pt_ = calibrate(sv[0], y_val, st[0])
            Ov = dict(O["val"], ps=pv, us=sv[1])
            Ot = dict(O["test"], ps=pt_, us=st[1])
            tau = Q.benchmark_ms(eo["sess_mc8_1t"], cfg.window, C, S, runs=100)
            netS = PL.net_costs(cfg, tau, eo["tau_det8"], C)
            PS = PL.tune_edgewarn(Ov, y_val, netS, cost, tune_masks)
            mt = PL.metrics(PL.edgewarn(Ot, up, up, PS, netS), y_te, events, g2l)
            rows_mc.append(dict(seed=seed, S=S, f1=mt["f1"], mean_uncertainty=float(st[1].mean()),
                                edge_ms=tau, escalated_pct=mt["escalated_pct"]))

        R = dict(auc=rows_auc[n0["auc"]:], r2=rows2[n0["r2"]:], r3=rows3[n0["r3"]:], r4=rows4[n0["r4"]:], mc=rows_mc[n0["mc"]:],
                 f2=rows_f2[n0["f2"]:], f3=rows_f3[n0["f3"]:],
                 sizes={k: v[sizes_before.get(k, 0):] for k, v in sizes.items()},
                 basin_f1={k: v[bf_before.get(k, 0):] for k, v in basin_f1.items()})
        pickle.dump(R, open(ck, "wb"))
        pd.DataFrame(rows_auc).to_csv(os.path.join(cfg.out_dir, "auc_raw.csv"), index=False)
        write_outputs(cfg, rows2, rows3, rows4, rows_mc, rows_f2, rows_f3, sizes, basin_f1)
        log(f"  seed {seed} saved; tables updated")

    pd.DataFrame(rows_auc).to_csv(os.path.join(cfg.out_dir, "auc_raw.csv"), index=False)
    write_outputs(cfg, rows2, rows3, rows4, rows_mc, rows_f2, rows_f3, sizes, basin_f1)
    log(f"Done. Results in {cfg.out_dir}/")


def ms(df, cols, by):
    g = df.groupby(by, sort=False)[cols]
    m, s = g.mean(), g.std().fillna(0)
    out = m.copy().astype(object)
    for c in cols:
        out[c] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(m[c], s[c])]
    return out


def write_outputs(cfg, rows2, rows3, rows4, rows_mc, rows_f2, rows_f3, sizes, basin_f1):
    from scipy.stats import wilcoxon
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    od = cfg.out_dir
    t2, t3, t4 = pd.DataFrame(rows2), pd.DataFrame(rows3), pd.DataFrame(rows4)
    tmc, f2, f3 = pd.DataFrame(rows_mc), pd.DataFrame(rows_f2), pd.DataFrame(rows_f3)
    for name, df in (("table2_raw", t2), ("table3_raw", t3), ("table4_raw", t4), ("table5_mc_raw", tmc),
                     ("fig2_tradeoff", f2), ("fig3_availability", f3)):
        df.to_csv(os.path.join(od, f"{name}.csv"), index=False)

    md = [f"# EdgeWarn results ({cfg.data_source} data, seeds {list(cfg.seeds)})\n"]
    if cfg.data_source == "synthetic":
        md.append("> **Synthetic data: pipeline check only. Do not report these numbers.**\n")

    T2 = ms(t2, ["precision", "recall", "f1", "far", "lead_h"], "method")
    md.append(f"## Table 2 Prediction quality (H = {cfg.horizon} days, no link failure)\n")
    md.append(T2.to_markdown())

    # Wilcoxon: EdgeWarn vs each baseline, per-basin F1 averaged over seeds
    md.append("\n\nWilcoxon signed-rank test on per-basin F1 (EdgeWarn vs method):\n")
    basins = sorted({b for (_, b) in basin_f1})
    ew = np.array([np.mean(basin_f1[("EdgeWarn", b)]) for b in basins])
    for m in t2["method"].unique():
        if m == "EdgeWarn":
            continue
        other = np.array([np.mean(basin_f1[(m, b)]) for b in basins])
        if len(basins) >= 6 and np.any(ew != other):
            p = wilcoxon(ew, other).pvalue
            md.append(f"- {m}: median ΔF1 = {np.median(ew - other):+.3f}, p = {p:.4f}")
        else:
            md.append(f"- {m}: too few basins or identical results for a test")

    T3 = ms(t3, ["latency_ms", "energy_mJ", "escalated_pct"], "method")
    T3.insert(0, "model size (kB)", [f"{np.mean(sizes[m]):.1f}" for m in T3.index])
    md.append("\n\n## Table 3 System cost per sensing window\n")
    md.append(T3.to_markdown())
    md.append(f"\nEdge inference on THIS machine: deterministic {np.mean(sizes['_edge_ms_det']):.3f} ms, "
              f"MC (S = {cfg.mc_samples}) {np.mean(sizes['_edge_ms_mc']):.3f} ms. "
              f"Student INT8 {np.mean(sizes['EdgeWarn']):.1f} kB vs FP32 {np.mean(sizes['_student_fp32_kb']):.1f} kB. "
              f"Parameters: teacher {int(np.mean(sizes['_params_teacher']))}, fog {int(np.mean(sizes['_params_fog']))}, "
              f"student {int(np.mean(sizes['_params_student']))}.\n"
              "Re-measure edge time and energy on the gateway with bench_gateway.py.")

    T4 = ms(t4, ["f1", "lead_h", "energy_mJ", "availability"], "variant")
    md.append(f"\n\n## Table 4 Ablation (hazard-correlated failures, rho = {cfg.ablation_rho})\n")
    md.append(T4.to_markdown())

    T5 = ms(tmc, ["f1", "mean_uncertainty", "edge_ms", "escalated_pct"], "S")
    md.append("\n\n## Table 5 Effect of Monte Carlo samples S\n")
    md.append(T5.to_markdown())

    A = f3.groupby(["kind", "rho", "method"])["availability"].agg(["mean", "std"]).reset_index()
    md.append("\n\n## Fig. 3 data: warning availability (%) vs link failure rate\n")
    md.append(A.pivot_table(index=["kind", "rho"], columns="method", values="mean").round(1).to_markdown())

    with open(os.path.join(od, "tables.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    # ---- figures (journal style: Arial, no title inside the figure)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for m, mk in (("EdgeWarn", "o"), ("Fixed-confidence offloading", "s")):
        d = f2[f2.method == m].groupby("knob")[["escalated_pct", "f1"]].mean().sort_values("escalated_pct")
        ax.plot(d["escalated_pct"], d["f1"], marker=mk, label=m)
    ax.set_xlabel("Windows escalated (%)")
    ax.set_ylabel("F1-score")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(od, "Fig2.png"), dpi=600)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    styles = {"Cloud-only (teacher)": "--", "Edge-only (student)": ":", "Fixed-confidence offloading": "-.",
              "EdgeWarn": "-"}
    for ax, kind, lab in zip(axs, ("independent", "correlated"), ("(a)", "(b)")):
        for m, ls in styles.items():
            d = A[(A.kind == kind) & (A.method == m)]
            ax.errorbar(d["rho"], d["mean"], yerr=d["std"], ls=ls, marker="o", ms=3, capsize=2, label=m)
        ax.set_xlabel("Link failure rate ρ")
        ax.text(0.02, 0.04, f"{lab} {kind} failures", transform=ax.transAxes)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Warning availability (%)")
    axs[1].legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 0.1))
    fig.tight_layout()
    fig.savefig(os.path.join(od, "Fig3.png"), dpi=600)


if __name__ == "__main__":
    main()
