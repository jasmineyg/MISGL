# coding=utf-8

"""Utilities for selecting an idle NVIDIA GPU before training starts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import logging
import os
import subprocess
import time


_GPU_LOCK_HANDLES = []


class GpuInfo(object):
  def __init__(self, index, memory_used_mb, memory_total_mb, utilization_pct):
    self.index = int(index)
    self.memory_used_mb = int(memory_used_mb)
    self.memory_total_mb = int(memory_total_mb)
    self.utilization_pct = int(utilization_pct)

  @property
  def memory_free_mb(self):
    return self.memory_total_mb - self.memory_used_mb

  def summary(self):
    return 'gpu {}: mem {}/{} MB, util {}%'.format(
      self.index, self.memory_used_mb, self.memory_total_mb, self.utilization_pct
    )


def _release_gpu_locks():
  while _GPU_LOCK_HANDLES:
    handle = _GPU_LOCK_HANDLES.pop()
    try:
      handle.close()
    except Exception:
      pass


atexit.register(_release_gpu_locks)


def _parse_int(value):
  return int(str(value).strip())


def _normalize_candidates(candidate_devices):
  if candidate_devices is None:
    return None
  if isinstance(candidate_devices, str):
    raw_items = candidate_devices.split(',')
  else:
    raw_items = candidate_devices

  candidates = []
  for item in raw_items:
    s = str(item).strip()
    if not s or s.lower() in ('all', 'auto', 'none', 'null'):
      continue
    candidates.append(_parse_int(s))
  return set(candidates) if candidates else None


def query_nvidia_gpus(nvidia_smi_path='nvidia-smi'):
  cmd = [
    nvidia_smi_path,
    '--query-gpu=index,memory.used,memory.total,utilization.gpu',
    '--format=csv,noheader,nounits',
  ]
  try:
    completed = subprocess.run(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      universal_newlines=True,
      check=True,
    )
  except FileNotFoundError:
    raise RuntimeError('nvidia-smi was not found. Automatic GPU selection requires NVIDIA drivers.')
  except subprocess.CalledProcessError as exc:
    stderr = exc.stderr.strip() if exc.stderr else ''
    raise RuntimeError('nvidia-smi failed while selecting a GPU. {}'.format(stderr))

  gpus = []
  for line in completed.stdout.splitlines():
    line = line.strip()
    if not line:
      continue
    fields = [field.strip() for field in line.split(',')]
    if len(fields) != 4:
      raise RuntimeError('Unexpected nvidia-smi output line: {}'.format(line))
    gpus.append(GpuInfo(
      index=fields[0],
      memory_used_mb=fields[1],
      memory_total_mb=fields[2],
      utilization_pct=fields[3],
    ))

  if not gpus:
    raise RuntimeError('nvidia-smi returned no GPUs.')
  return gpus


def _try_acquire_gpu_lock(gpu_index, lock_dir):
  if os.name != 'posix':
    return None
  try:
    import fcntl
  except ImportError:
    return None

  os.makedirs(lock_dir, exist_ok=True)
  lock_path = os.path.join(lock_dir, 'gpu_{}.lock'.format(gpu_index))
  handle = open(lock_path, 'a+')
  try:
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except OSError:
    handle.close()
    return None

  handle.seek(0)
  handle.truncate()
  handle.write('pid={}\n'.format(os.getpid()))
  handle.flush()
  _GPU_LOCK_HANDLES.append(handle)
  return handle


def _gpu_is_idle(gpu, memory_used_max_mb, utilization_max_pct):
  if memory_used_max_mb is not None and gpu.memory_used_mb > int(memory_used_max_mb):
    return False
  if utilization_max_pct is not None and gpu.utilization_pct > int(utilization_max_pct):
    return False
  return True


def _format_gpu_list(gpus):
  return '; '.join(gpu.summary() for gpu in gpus)


def select_idle_gpu(memory_used_max_mb=1024,
                    utilization_max_pct=10,
                    wait_seconds=0,
                    poll_interval_seconds=30,
                    candidate_devices=None,
                    nvidia_smi_path='nvidia-smi',
                    lock=True,
                    lock_dir='/tmp/misgl_gpu_locks',
                    logger=None):
  """Return one idle physical GPU index as a string.

  The selected index is suitable for CUDA_VISIBLE_DEVICES. On Linux, the
  optional file lock reduces the chance that two MISGL jobs choose the same idle
  GPU at the same time.
  """
  logger = logger or logging.getLogger(__name__)
  candidates = _normalize_candidates(candidate_devices)
  wait_seconds = int(wait_seconds or 0)
  poll_interval_seconds = max(1, int(poll_interval_seconds or 30))
  deadline = time.time() + wait_seconds if wait_seconds > 0 else None

  while True:
    gpus = query_nvidia_gpus(nvidia_smi_path=nvidia_smi_path)
    if candidates is not None:
      gpus = [gpu for gpu in gpus if gpu.index in candidates]
      if not gpus:
        raise RuntimeError('No GPUs matched candidate devices: {}'.format(sorted(candidates)))

    idle_gpus = [
      gpu for gpu in gpus
      if _gpu_is_idle(gpu, memory_used_max_mb, utilization_max_pct)
    ]
    idle_gpus.sort(key=lambda gpu: (-gpu.memory_free_mb, gpu.utilization_pct, gpu.memory_used_mb, gpu.index))

    for gpu in idle_gpus:
      if lock:
        lock_handle = _try_acquire_gpu_lock(gpu.index, lock_dir)
        if lock_handle is None and os.name == 'posix':
          continue
      logger.warning('Auto selected GPU {} ({})'.format(gpu.index, gpu.summary()))
      return str(gpu.index)

    status = _format_gpu_list(gpus)
    if deadline is None or time.time() >= deadline:
      raise RuntimeError(
        'No idle GPU found. Thresholds: memory_used <= {} MB, utilization <= {}%. Current: {}'.format(
          memory_used_max_mb, utilization_max_pct, status
        )
      )

    remaining = max(0, deadline - time.time())
    sleep_seconds = min(poll_interval_seconds, remaining)
    logger.warning('No idle GPU found yet. Current: {}. Retry in {:.0f}s.'.format(status, sleep_seconds))
    time.sleep(sleep_seconds)
