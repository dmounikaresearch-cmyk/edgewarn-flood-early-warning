"""Experiment configuration for EdgeWarn.

Values marked ASSUMED are placeholders. Replace them with values you measure
on your own hardware or cite from the literature before reporting results.
"""
from dataclasses import dataclass, asdict
import json


@dataclass
class Config:
    # ---------------- data ----------------
    data_source: str = "synthetic"       # "synthetic" or "camels"
    camels_root: str = ""                # folder containing basin_mean_forcing/, usgs_streamflow/, camels_topo.txt
    basins_file: str = ""                # optional text file, one 8-digit gauge id per line
    max_basins: int = 50                 # cap for CAMELS runs (memory/time)
    n_synth_basins: int = 20
    n_synth_days: int = 3000
    window: int = 30                     # W, days of history per window
    horizon: int = 2                     # H, days ahead
    exceed_pct: float = 95.0             # flood threshold percentile (training period)
    n_neighbours: int = 3                # K neighbouring gateways seen by fog/teacher
    train_frac: float = 0.6              # chronological split
    val_frac: float = 0.2

    # ---------------- models ----------------
    teacher_ch: int = 64
    teacher_window: int = 365            # cloud model sees one year of history
    fog_window: int = 180
    teacher_hidden: int = 64
    teacher_samples_per_epoch: int = 40000
    epochs_cloud: int = 8
    fog_hidden: int = 48
    teacher_lr: float = 5e-4
    fog_ch: int = 32
    student_ch: int = 16
    rnn_hidden: int = 32
    dropout: float = 0.2
    epochs_teacher: int = 10
    epochs_student: int = 10
    batch: int = 256
    samples_per_epoch: int = 60000      # random training windows drawn each epoch (0 = all)
    lr: float = 1e-3
    kd_alpha: float = 0.5
    kd_T: float = 2.0
    mc_samples: int = 10                 # S, Monte Carlo dropout passes

    # ---------------- cost function (Sec. 3.4) ----------------
    cost_miss: float = 10.0
    cost_fa: float = 1.0
    lambda_lat: float = 1e-4             # cost per ms
    lambda_energy: float = 1e-3          # cost per mJ
    latency_max_ms: float = 5000.0

    # ---------------- network & power (ASSUMED) ----------------
    bw_gf_kbps: float = 50.0             # gateway->fog bandwidth (LoRaWAN/NB-IoT class)
    bw_fc_kbps: float = 10000.0          # fog->cloud bandwidth
    rtt_gf_ms: float = 300.0             # gateway<->fog round trip
    rtt_fc_ms: float = 80.0              # fog<->cloud round trip
    fog_ms: float = 15.0                 # fog compute per window (replace with measured)
    cloud_ms: float = 5.0                # cloud compute per window (replace with measured)
    p_cpu_w: float = 2.5                 # gateway active power above idle (measure with USB meter)
    p_tx_w: float = 1.0                  # radio power while transmitting
    bytes_per_value: int = 4

    # ---------------- failures & evaluation ----------------
    failure_rates: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    ablation_rho: float = 0.3
    tune_rho: float = 0.3                # failure rate used to tune theta_safe
    seeds: tuple = (0, 1, 2, 3, 4)
    data_seed: int = 0
    out_dir: str = "results"
    quick: bool = False

    def apply_quick(self):
        """Small settings for a fast end-to-end check (not for the paper)."""
        self.n_synth_basins = 10
        self.n_synth_days = 1600
        self.epochs_teacher = 3
        self.epochs_student = 3
        self.epochs_cloud = 2
        self.teacher_window = 60
        self.fog_window = 45
        self.teacher_samples_per_epoch = 3000
        self.seeds = (0, 1)
        self.mc_samples = 5
        self.max_basins = 5
        return self

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
