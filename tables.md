# EdgeWarn results (camels data, seeds [0, 1, 2])

## Table 2 Prediction quality (H = 2 days, no link failure)

| method                      | precision     | recall        | f1            | far           | lead_h         |
|:----------------------------|:--------------|:--------------|:--------------|:--------------|:---------------|
| Cloud-only (teacher)        | 0.300 ± 0.011 | 0.742 ± 0.008 | 0.427 ± 0.010 | 0.160 ± 0.010 | 38.074 ± 0.090 |
| Edge-only (student)         | 0.302 ± 0.029 | 0.745 ± 0.024 | 0.429 ± 0.024 | 0.161 ± 0.026 | 37.635 ± 0.975 |
| Edge LSTM                   | 0.306 ± 0.017 | 0.747 ± 0.012 | 0.434 ± 0.015 | 0.158 ± 0.015 | 37.464 ± 0.466 |
| Edge GRU                    | 0.303 ± 0.013 | 0.756 ± 0.008 | 0.433 ± 0.012 | 0.161 ± 0.012 | 37.668 ± 0.453 |
| Edge TCN (no distillation)  | 0.292 ± 0.020 | 0.751 ± 0.015 | 0.420 ± 0.019 | 0.170 ± 0.020 | 38.343 ± 0.529 |
| Fixed-confidence offloading | 0.307 ± 0.035 | 0.744 ± 0.028 | 0.433 ± 0.029 | 0.158 ± 0.030 | 37.641 ± 1.155 |
| EdgeWarn                    | 0.280 ± 0.005 | 0.763 ± 0.003 | 0.410 ± 0.005 | 0.181 ± 0.005 | 38.313 ± 0.145 |


Wilcoxon signed-rank test on per-basin F1 (EdgeWarn vs method):

- Cloud-only (teacher): median ΔF1 = -0.019, p = 0.0000
- Edge-only (student): median ΔF1 = -0.018, p = 0.0000
- Edge LSTM: median ΔF1 = -0.024, p = 0.0000
- Edge GRU: median ΔF1 = -0.022, p = 0.0000
- Edge TCN (no distillation): median ΔF1 = -0.009, p = 0.0000
- Fixed-confidence offloading: median ΔF1 = -0.022, p = 0.0000


## Table 3 System cost per sensing window

| method                      |   model size (kB) | latency_ms      | energy_mJ      | escalated_pct   |
|:----------------------------|------------------:|:----------------|:---------------|:----------------|
| Cloud-only (teacher)        |             191.3 | 481.480 ± 0.000 | 96.000 ± 0.000 | 100.000 ± 0.000 |
| Edge-only (student)         |              54.1 | 0.031 ± 0.005   | 0.078 ± 0.014  | 0.000 ± 0.000   |
| Fixed-confidence offloading |              54.1 | 11.138 ± 2.911  | 2.292 ± 0.586  | 2.307 ± 0.604   |
| EdgeWarn                    |              54.1 | 0.391 ± 0.062   | 0.957 ± 0.139  | 0.002 ± 0.003   |

Edge inference on THIS machine: deterministic 0.031 ms, MC (S = 10) 0.382 ms. Student INT8 54.1 kB vs FP32 44.2 kB. Parameters: teacher 48961, fog 20353, student 5857.
Re-measure edge time and energy on the gateway with bench_gateway.py.


## Table 4 Ablation (hazard-correlated failures, rho = 0.3)

| variant                                       | f1            | lead_h         | energy_mJ     | availability   |
|:----------------------------------------------|:--------------|:---------------|:--------------|:---------------|
| EdgeWarn (full)                               | 0.410 ± 0.004 | 38.313 ± 0.125 | 0.955 ± 0.119 | 70.599 ± 0.541 |
| Without distillation                          | 0.423 ± 0.009 | 38.243 ± 0.330 | 0.962 ± 0.112 | 67.174 ± 1.261 |
| Without quantization                          | 0.435 ± 0.005 | 37.529 ± 0.288 | 0.864 ± 0.120 | 66.780 ± 0.906 |
| Without hazard-dependent threshold (beta = 0) | 0.410 ± 0.004 | 38.313 ± 0.125 | 0.955 ± 0.119 | 70.599 ± 0.541 |
| Without link-aware fallback                   | 0.410 ± 0.004 | 38.313 ± 0.125 | 0.955 ± 0.119 | 70.599 ± 0.541 |


## Table 5 Effect of Monte Carlo samples S

|   S | f1            | mean_uncertainty   | edge_ms       | escalated_pct   |
|----:|:--------------|:-------------------|:--------------|:----------------|
|   1 | 0.409 ± 0.000 | 0.000 ± 0.000      | 0.102 ± 0.000 | 0.000 ± 0.000   |
|   5 | 0.413 ± 0.000 | 0.026 ± 0.000      | 0.242 ± 0.000 | 0.000 ± 0.000   |
|  10 | 0.414 ± 0.000 | 0.028 ± 0.000      | 0.454 ± 0.000 | 0.051 ± 0.000   |
|  20 | 0.428 ± 0.000 | 0.029 ± 0.000      | 0.783 ± 0.000 | 0.000 ± 0.000   |


## Fig. 3 data: warning availability (%) vs link failure rate

|                      |   Cloud-only (teacher) |   Edge-only (student) |   EdgeWarn |   Fixed-confidence offloading |
|:---------------------|-----------------------:|----------------------:|-----------:|------------------------------:|
| ('correlated', 0.0)  |                   66.1 |                  67.5 |       70.6 |                          67   |
| ('correlated', 0.1)  |                   53.4 |                  67.5 |       70.6 |                          66.9 |
| ('correlated', 0.2)  |                   41.7 |                  67.5 |       70.6 |                          67   |
| ('correlated', 0.3)  |                   34.6 |                  67.5 |       70.6 |                          67   |
| ('correlated', 0.4)  |                   29.9 |                  67.5 |       70.6 |                          67   |
| ('correlated', 0.5)  |                   26.4 |                  67.5 |       70.6 |                          67   |
| ('independent', 0.0) |                   66.1 |                  67.5 |       70.6 |                          67   |
| ('independent', 0.1) |                   59.3 |                  67.5 |       70.6 |                          66.9 |
| ('independent', 0.2) |                   50.5 |                  67.5 |       70.6 |                          66.8 |
| ('independent', 0.3) |                   41.4 |                  67.5 |       70.6 |                          66.9 |
| ('independent', 0.4) |                   32.2 |                  67.5 |       70.6 |                          66.9 |
| ('independent', 0.5) |                   22.9 |                  67.5 |       70.6 |                          66.8 |
