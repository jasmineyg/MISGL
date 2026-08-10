#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/yg/Subgraph-MIL/diffpool2
PYTHON=/data/yg/Anaconda/anaconda3/envs/diffpool/bin/python
RUN_ROOT="$ROOT/results/loss_ablation/gat_mil_head_20260623"
LOG_DIR="$RUN_ROOT/logs"
PID_DIR="$RUN_ROOT/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"
cd "$ROOT"

sha256sum   MISGL/models/encoder.py   MISGL/models/mil_head.py   MISGL/utils/get_loss.py   MISGL/utils/evaluate.py   MISGL/utils/hparams_lib.py   MISGL/bin/train_eval.py   config/loss_ablation_bce.yml   config/loss_ablation_focal.yml   config/loss_ablation_weighted_bce.yml   > "$RUN_ROOT/code_sha256.txt"
{
  date --iso-8601=seconds
  "$PYTHON" --version
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
} > "$RUN_ROOT/run_manifest.txt"

launch_one() {
  local loss_name="$1"
  local dataset="$2"
  local gpu="$3"
  local config="config/loss_ablation_${loss_name}.yml"
  local name="${dataset}_${loss_name}"
  local log="$LOG_DIR/${name}.log"
  local pid_file="$PID_DIR/${name}.pid"

  if [[ -f "$pid_file" ]]; then
    local old_pid
    old_pid=$(cat "$pid_file")
    if kill -0 "$old_pid" 2>/dev/null; then
      echo "already running: $name pid=$old_pid"
      return
    fi
  fi

  nohup "$PYTHON" train.py     --hparam_path "$config"     --data_name_set "$dataset"     --auto_select_gpu     --gpu_candidate_devices "$gpu"     > "$log" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$pid_file"
  echo "started: $name pid=$pid gpu=$gpu log=$log"
}

launch_one bce ogbn_arxiv 0
launch_one focal ogbn_arxiv 1
launch_one weighted_bce ogbn_arxiv 2
launch_one bce reddit 3
launch_one focal reddit 4
launch_one weighted_bce reddit 5
