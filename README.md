# EdgeWarn

Code for the manuscript *Edge-first flood early warning on IoT gateways: compact
uncertainty-aware models and resilience to hazard-correlated link failures*.

It trains the teacher, fog and student models, distils and quantizes the student to INT8 ONNX,
runs the hazard- and link-aware offloading policy against cloud-only, edge-only and
fixed-confidence baselines, simulates independent and hazard-correlated link failures, and
writes every table and figure in Section 6 of the manuscript.

## Install

```bash
pip install torch onnx onnxruntime numpy pandas scipy scikit-learn matplotlib tabulate
```

## Quick start

```bash
bash run_all.sh --quick                                   # 10-minute pipeline check (synthetic data)
bash run_all.sh /path/to/basin_dataset_public_v1p2        # full CAMELS run, 50 basins, seeds 0 1 2
```

`results_camels50/` holds the per-seed results reported in the paper (3 seeds).
`basins_50.txt` lists the 50 CAMELS gauges (HUC 01-02) used in the paper. Re-running the same
command resumes from the last finished seed.

## Models

| Tier | Model | Input | Notes |
| --- | --- | --- | --- |
| Edge (gateway) | TCN student, 16 channels, ~5.9k parameters | own basin, last 30 days | distilled from the cloud model, INT8 ONNX, MC dropout |
| Fog | LSTM (48) + neighbour attention | own basin 180 days + 3 neighbours 30 days | |
| Cloud | LSTM (64) + neighbour attention | own basin 365 days + 3 neighbours 30 days | teacher for distillation |

## 1. Check the pipeline (synthetic data, about 10 min on a laptop CPU)

```bash
python run_experiments.py --quick --out results_quick
```

The synthetic generator exists only to test the code. **Never report its numbers.**

## 2. Run on CAMELS-US

1. Download *CAMELS time series meteorology, observed flow, meta data (v1.2)* from the NCAR/UCAR
   CAMELS page and unzip it. You need the folder that contains `basin_mean_forcing/`,
   `usgs_streamflow/` and `camels_topo.txt` (copy `camels_topo.txt` in from the attributes download).
2. Optionally create your own `basins.txt` with one 8-digit gauge ID per line (e.g. 50 basins in one region,
   so neighbouring gateways are geographically meaningful).
3. Run:

```bash
python run_experiments.py --source camels --camels_root /path/to/basin_dataset_public_v1p2 \
       --basins_file basins.txt --max_basins 50 --out results_camels
```

Each seed takes about one hour on a 2-vCPU machine for 50 basins; use a GPU machine or
fewer basins first. Check the file paths in `edgewarn/data.py` (`load_camels`) against your download.

## Correctness notes (read before comparing with earlier outputs)

- **Calendar alignment.** `load_camels` keeps only the dates present in every selected basin
  (1986-07-18 to 2014-12-31 for `basins_50.txt`) and stacks basins on those dates, so day *t* is
  the same calendar day in every gateway and its neighbours. An earlier version sliced each basin
  from its own start date; results from that version are invalid.
- **Equal threshold grid.** Every method (cloud, edge, baselines, fixed-confidence, EdgeWarn)
  tunes its decision threshold over the same 37-point grid `THETA_GRID` in `edgewarn/policy.py`.
- **Threshold-free metrics.** `auc_raw.csv` gives PR-AUC and ROC-AUC per model and seed;
  `predictions_seed*.npz` stores all test/validation probabilities so any metric can be recomputed
  without retraining.

## 3. Measure on the gateway

Copy `results_camels/models/student_s0_mc_int8.onnx` and `student_s0_det_int8.onnx` to the
Raspberry Pi and follow the instructions at the top of `bench_gateway.py`. Put the measured power
into `edgewarn/config.py` (`p_cpu_w`) and re-run step 2. The edge timings written by
`run_experiments.py` come from the machine it runs on, not the gateway.

## Values you must replace before reporting

All entries marked **ASSUMED** in `edgewarn/config.py`: link bandwidths and round-trip times,
fog and cloud compute times, gateway CPU and radio power. Cite a source or measure each one.

## Outputs (`results_*/`)

| File | Manuscript item |
| --- | --- |
| `tables.md` | Tables 2-5 (mean ± SD over seeds), Wilcoxon tests, Fig. 3 data |
| `Fig2.png` | F1 vs fraction escalated |
| `Fig3.png` | Warning availability vs link failure rate |
| `*_raw.csv` | Per-seed values behind every table |
| `thresholds_seed*.json` | Tuned thresholds |
| `auc_raw.csv` | PR-AUC / ROC-AUC per model and seed |
| `predictions_seed*.npz` | Test and validation probabilities for every model |
| `models/*.onnx` | FP32 and INT8 student models |

## Figures

```bash
cd figures
python make_figs.py ../results_camels50 .                 # Fig2.png, Fig3.png
mmdc -i fig1.mmd -o Fig1.png -c mcfg.json -s 4           # Fig1.png (architecture)
```


## Definitions used in the code

- Window label: 1 if streamflow exceeds the basin's training-period 95th percentile within the next H days.
- False-alarm rate: FP / (FP + TN) over windows.
- Event: a flood onset (first exceedance day). An event counts as warned if any window in the H days
  before onset issues a warning; lead time = days between the earliest such warning and onset, × 24 h.
- Warning availability: warned events / all events in the test period.
- Hazard-correlated failures: per-window failure probability proportional to 3-day rainfall, scaled
  so its mean equals ρ (capped at 0.95).
- The network is simulated in Python (`edgewarn/policy.py`), not iFogSim.

## Citation

If you use this code, please cite: D. Mounika and J. Kolluri, *Edge-first flood early warning on IoT
gateways: compact uncertainty-aware models and resilience to hazard-correlated link failures*
(manuscript submitted to Discover Artificial Intelligence).

## License

MIT (see `LICENSE`).
