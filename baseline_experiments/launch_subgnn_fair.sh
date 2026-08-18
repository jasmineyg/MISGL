#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU_ID DATASET_KEY [DATASET_KEY ...]" >&2
  exit 2
fi

gpu_id="$1"
shift
repo="/data/yg/Subgraph-MIL/SubGNN2/SubGNN"
manifest="/data/yg/Subgraph-MIL/diffpool2/results/paper_10fold_20260814/execution_manifest.json"
result_root="/data/yg/Subgraph-MIL/SubGNN2/results/fair_10fold_20260815"
cache_root="/data/yg/Subgraph-MIL/SubGNN2/dataset_cache/fair_10fold_20260815"
python_bin="/data/yg/Anaconda/anaconda3/envs/SubGNN/bin/python"

cd "$repo"
for dataset_key in "$@"; do
  echo "[$(date --iso-8601=seconds)] start $dataset_key on physical GPU $gpu_id"
  CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" -u fair_experiment.py \
    --execution-manifest "$manifest" \
    --dataset-key "$dataset_key" \
    --result-root "$result_root" \
    --cache-root "$cache_root" \
    --device cuda:0
  echo "[$(date --iso-8601=seconds)] complete $dataset_key"
done
