## Automatic GPU selection

Training can choose an idle NVIDIA GPU on the Linux server where `train.py` is
running. Enable it either from the command line:

```bash
python train.py --hparam_path ./config/b_on.yml --auto_select_gpu
```

or in a YAML config:

```yaml
device: 'cuda'
cuda_visible_devices: 'auto'
# Alternatively leave cuda_visible_devices unchanged and set:
# auto_select_gpu: true
gpu_candidate_devices: null       # e.g. '0,1,2'; null scans all visible GPUs
gpu_memory_used_max_mb: 1024      # max used memory for an idle card
gpu_utilization_max_pct: 10       # max utilization for an idle card
gpu_select_wait_seconds: 0        # wait time before failing; 0 fails immediately
gpu_select_poll_interval: 30
gpu_lock_idle_card: true          # Linux file lock to reduce duplicate selection
gpu_lock_dir: '/tmp/misgl_gpu_locks'
```

The selector uses `nvidia-smi` and sets `CUDA_VISIBLE_DEVICES` before CUDA is
initialized. When automatic selection is enabled, use `gpu_candidate_devices`
to restrict which physical cards may be selected. Manual
`cuda_visible_devices: '0'` style settings still work when automatic selection
is disabled.
