#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/yg/Subgraph-MIL/diffpool2
PYTHON=/data/yg/Anaconda/anaconda3/envs/diffpool/bin/python
RUN_ROOT="$ROOT/results/subgnn_border"
LOG_DIR="$RUN_ROOT/logs"
SUMMARY_DIR="$RUN_ROOT/summary"
mkdir -p "$LOG_DIR" "$SUMMARY_DIR"
cd "$ROOT"

run_config() {
  local name="$1"
  local cfg="config/subgnn_border_${name}.yml"
  local log="$LOG_DIR/${name}.log"
  echo "===== ${name} $(date '+%F %T') =====" | tee "$log"
  "$PYTHON" train.py --hparam_path "$cfg" --data_name_set ogbn_arxiv reddit 2>&1 | tee -a "$log"
}

run_config route
run_config shuffled

"$PYTHON" summarize_subgnn_border.py \
  --root results/subgnn_border \
  --output results/subgnn_border/summary/subgnn_border_report.md \
  2>&1 | tee "$LOG_DIR/summary.log"
