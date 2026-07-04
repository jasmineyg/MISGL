#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/yg/Subgraph-MIL/diffpool2
PYTHON=/data/yg/Anaconda/anaconda3/envs/diffpool/bin/python
RUN_ROOT="$ROOT/results/subgnn_border"
LOG_DIR="$RUN_ROOT/logs"
SUMMARY_DIR="$RUN_ROOT/summary"
mkdir -p "$LOG_DIR" "$SUMMARY_DIR"
cd "$ROOT"

sha256sum \
  train.py \
  summarize_subgnn_border.py \
  MISGL/utils/subgnn_border.py \
  MISGL/utils/load_data.py \
  MISGL/utils/hparams_lib.py \
  MISGL/utils/global_variables.py \
  MISGL/models/subgnn_border.py \
  MISGL/models/encoder.py \
  MISGL/bin/train_eval.py \
  config/subgnn_border_baseline.yml \
  config/subgnn_border_route.yml \
  config/subgnn_border_shuffled.yml \
  tests/test_subgnn_border.py \
  > "$RUN_ROOT/code_sha256.txt"

run_config() {
  local name="$1"
  local cfg="config/subgnn_border_${name}.yml"
  local log="$LOG_DIR/${name}.log"
  echo "===== ${name} $(date '+%F %T') =====" | tee "$log"
  "$PYTHON" train.py --hparam_path "$cfg" --data_name_set ogbn_arxiv reddit 2>&1 | tee -a "$log"
}

run_config baseline
run_config route
run_config shuffled

"$PYTHON" summarize_subgnn_border.py \
  --root results/subgnn_border \
  --output results/subgnn_border/summary/subgnn_border_report.md \
  2>&1 | tee "$LOG_DIR/summary.log"
