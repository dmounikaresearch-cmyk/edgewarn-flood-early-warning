#!/usr/bin/env bash
# EdgeWarn: full pipeline in one command.
#
# Usage:
#   bash run_all.sh /path/to/basin_dataset_public_v1p2            # CAMELS-US, 50 basins, 3 seeds
#   bash run_all.sh /path/to/basin_dataset_public_v1p2 0 1 2 3 4  # choose seeds
#   bash run_all.sh --quick                                       # 10-minute check on synthetic data
#
# The CAMELS folder is the one that contains basin_mean_forcing/, usgs_streamflow/ and basin_metadata/
# (from basin_timeseries_v1p2_metForcing_obsFlow.zip, https://gdex.ucar.edu/dataset/camels.html).
# If the run stops (e.g. a time limit), run the same command again: finished seeds are reloaded.
set -euo pipefail
cd "$(dirname "$0")"

pip install -q torch onnx onnxruntime numpy pandas scipy scikit-learn matplotlib tabulate

if [[ "${1:-}" == "--quick" ]]; then
  python run_experiments.py --quick --out results_quick
  exit 0
fi

ROOT="${1:?give the CAMELS folder (basin_dataset_public_v1p2)}"
shift
SEEDS="${*:-0 1 2}"
BASINS="basins_50.txt"   # 50 basins from HUC regions 01-02 (New England / Mid-Atlantic) used in the paper

python run_experiments.py --source camels --camels_root "$ROOT" --basins_file "$BASINS" \
       --max_basins 50 --seeds $SEEDS --out results_camels50 2>&1 | tee -a results_camels50.log

echo "Tables:  results_camels50/tables.md"
echo "Figures: results_camels50/Fig2.png, results_camels50/Fig3.png"
echo "Gateway models to benchmark: results_camels50/models/student_s0_*_int8.onnx (see bench_gateway.py)"
