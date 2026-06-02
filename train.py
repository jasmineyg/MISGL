# coding=utf-8

# function main(args) and top-level imports
import os
import torch
import numpy as np
import logging
import argparse
import random
import time

from MISGL.utils import hparam
from MISGL.utils import hparams_lib
from MISGL.utils import reproducibility

_AUTO_GPU_VALUES = set(['auto', 'idle', 'free'])


def _parse_data_name_set(raw):
  if raw is None:
    return []
  if isinstance(raw, str):
    raw = [raw]
  names = []
  for item in raw:
    if item is None:
      continue
    s = str(item).strip()
    if not s:
      continue
    if ',' in s:
      parts = [p.strip() for p in s.split(',')]
      names.extend([p for p in parts if p])
    else:
      names.append(s)
  seen = set()
  deduped = []
  for n in names:
    if n not in seen:
      seen.add(n)
      deduped.append(n)
  return deduped


def _as_bool(value, default=False):
  if value is None:
    return default
  if isinstance(value, bool):
    return value
  if isinstance(value, (int, float)):
    return bool(value)
  s = str(value).strip().lower()
  if s in ('1', 'true', 'yes', 'y', 'on'):
    return True
  if s in ('0', 'false', 'no', 'n', 'off'):
    return False
  return default


def _arg_or_hparam(args, hparams, name, default=None):
  value = getattr(args, name, None)
  if value is not None:
    return value
  return getattr(hparams, name, default)


def _cuda_visible_devices_requests_auto(value):
  if value is None:
    return False
  return str(value).strip().lower() in _AUTO_GPU_VALUES


def _should_auto_select_gpu(args, hparams):
  if getattr(hparams, 'device', None) != 'cuda':
    return False
  arg_value = getattr(args, 'auto_select_gpu', None)
  if arg_value is not None:
    return bool(arg_value)
  if _as_bool(getattr(hparams, 'auto_select_gpu', False), default=False):
    return True
  return _cuda_visible_devices_requests_auto(getattr(hparams, 'cuda_visible_devices', None))


def _resolve_cuda_visible_devices(args, hparams):
  raw_cuda_visible_devices = getattr(hparams, 'cuda_visible_devices', None)

  if not _should_auto_select_gpu(args, hparams):
    if raw_cuda_visible_devices is not None:
      os.environ['CUDA_VISIBLE_DEVICES'] = str(raw_cuda_visible_devices)
    return None

  from MISGL.utils import gpu_auto_select

  selected_gpu = gpu_auto_select.select_idle_gpu(
    memory_used_max_mb=_arg_or_hparam(args, hparams, 'gpu_memory_used_max_mb', 1024),
    utilization_max_pct=_arg_or_hparam(args, hparams, 'gpu_utilization_max_pct', 10),
    wait_seconds=_arg_or_hparam(args, hparams, 'gpu_select_wait_seconds', 0),
    poll_interval_seconds=_arg_or_hparam(args, hparams, 'gpu_select_poll_interval', 30),
    candidate_devices=_arg_or_hparam(args, hparams, 'gpu_candidate_devices', None),
    nvidia_smi_path=_arg_or_hparam(args, hparams, 'nvidia_smi_path', 'nvidia-smi'),
    lock=_as_bool(getattr(hparams, 'gpu_lock_idle_card', True), default=True),
    lock_dir=getattr(hparams, 'gpu_lock_dir', '/tmp/misgl_gpu_locks'),
    logger=logging.getLogger(__name__),
  )
  hparams.cuda_visible_devices = selected_gpu
  os.environ['CUDA_VISIBLE_DEVICES'] = selected_gpu
  logging.warning('CUDA_VISIBLE_DEVICES={}'.format(selected_gpu))
  return selected_gpu


def main(args):

  base_hparams = hparam.HParams()
  base_hparams.from_yaml(args.hparam_path)
  hparams_lib.apply_defaults(base_hparams)
  if args.processed_data_dir:
    base_hparams.processed_data_dir = args.processed_data_dir

  selected_cuda_visible_devices = _resolve_cuda_visible_devices(args, base_hparams)

  # reproducibility
  reproducibility.set_seed(1024, cuda_deterministic=(getattr(base_hparams, 'device', None) == 'cuda'))

  from MISGL.bin import train_eval

  dataset_names = _parse_data_name_set(getattr(args, 'data_name_set', None))
  if not dataset_names:
    dataset_names = _parse_data_name_set(getattr(base_hparams, 'data_name_set', None))
  if not dataset_names:
    raise ValueError('No dataset specified. Please set data_name_set in the yaml, or pass --data_name_set on the command line.')

  all_dataset_summaries = []
  failed_datasets = []

  for idx, data_name in enumerate(dataset_names):
    hparams = hparam.HParams()
    hparams.from_yaml(args.hparam_path)
    hparams_lib.apply_defaults(hparams)
    hparams.data_name = data_name
    if selected_cuda_visible_devices is not None:
      hparams.cuda_visible_devices = selected_cuda_visible_devices
    if args.processed_data_dir:
      hparams.processed_data_dir = args.processed_data_dir

    if getattr(hparams, 'cuda_visible_devices', None) is not None:
      os.environ['CUDA_VISIBLE_DEVICES'] = str(hparams.cuda_visible_devices)

    if hparams.device == 'cuda':
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

    hparams.tb_unique_run_dir = False

    base_save_path = getattr(hparams, 'model_save_path', None)
    if base_save_path:
      hparams.model_save_path = os.path.join(base_save_path, data_name)
    else:
      hparams.model_save_path = os.path.join('results', data_name)
    os.makedirs(hparams.model_save_path, exist_ok=True)

    tb_root = getattr(hparams, 'tb_logdir', None)
    if tb_root:
      hparams.tb_logdir = os.path.join(tb_root, data_name)
    else:
      hparams.tb_logdir = os.path.join('..', 'result', data_name)

    base_ts = getattr(hparams, 'timestamp', None)
    base_ts = str(base_ts).strip() if base_ts is not None else ''
    if base_ts == '':
      base_ts = 'run'
    hparams.timestamp = f'{data_name}_{base_ts}'

    logging.warning('\n' + '='*30)
    logging.warning('==== {} ===='.format(data_name))
    logging.warning('='*30 + '\n')
    try:
      ret = train_eval.train_eval(hparams, data_name=data_name)
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
      summary = ret.get('summary', {}) if isinstance(ret, dict) else {}
      all_dataset_summaries.append({'data_name': data_name, 'summary': summary})
    except Exception as e:
      logging.exception('Dataset {} failed: {}'.format(data_name, str(e)))
      failed_datasets.append(data_name)

  if len(all_dataset_summaries) > 0 or failed_datasets:
    logging.warning('\n' + '='*60)
    logging.warning('===== All Datasets Summary =====')
    logging.warning('='*60)
    
    # Print header
    header = "{:<10} | {:<20} | {:<20} | {:<20} | {:<20}".format('Dataset', 'Accuracy', 'Precision', 'Recall', 'F1 Score')
    logging.warning(header)
    logging.warning('-'*len(header))

    for item in all_dataset_summaries:
      name = item.get('data_name')
      summary = item.get('summary', {}) or {}
      def _fmt(k):
        v = summary.get(k, None)
        if not isinstance(v, dict):
          return 'n/a'
        mean = v.get('mean', None)
        std = v.get('std', None)
        if mean is None or std is None:
          return 'n/a'
        return '{:.4f} ± {:.4f}'.format(float(mean), float(std))
      
      row = "{:<10} | {:<20} | {:<20} | {:<20} | {:<20}".format(
        name, _fmt('acc'), _fmt('prec'), _fmt('rec'), _fmt('F1')
      )
      logging.warning(row)
    
    logging.warning('='*60 + '\n')

    if failed_datasets:
      logging.warning('Failed datasets: {}'.format(', '.join(failed_datasets)))


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  parser = argparse.ArgumentParser(description='Parameters for the training of GNN')
  parser.add_argument('--hparam_path', nargs='?', type=str,
                      default='./config/hparams_testdb.yml',
                      help='The path to .yml file which contains all the hyperparameters.'
                      )
  parser.add_argument('--data_name_set', nargs='*', type=str, default=None,
                      help='Run multiple datasets in one shot, e.g. --data_name_set cs eng phy (also supports comma-separated string).')
  parser.add_argument('--processed_data_dir', nargs='?', type=str, default=None,
                      help='Override processed_data_dir from the yaml for the current run.')
  gpu_auto_group = parser.add_mutually_exclusive_group()
  gpu_auto_group.add_argument('--auto_select_gpu', dest='auto_select_gpu', action='store_true', default=None,
                              help='Automatically select an idle GPU with nvidia-smi before training.')
  gpu_auto_group.add_argument('--no_auto_select_gpu', dest='auto_select_gpu', action='store_false',
                              help='Disable automatic GPU selection even if the yaml requests it.')
  parser.add_argument('--gpu_candidate_devices', type=str, default=None,
                      help='Comma-separated physical GPU ids allowed for automatic selection, e.g. 0,1,3.')
  parser.add_argument('--gpu_memory_used_max_mb', type=int, default=None,
                      help='Maximum used GPU memory in MB for a GPU to be considered idle.')
  parser.add_argument('--gpu_utilization_max_pct', type=int, default=None,
                      help='Maximum GPU utilization percentage for a GPU to be considered idle.')
  parser.add_argument('--gpu_select_wait_seconds', type=int, default=None,
                      help='How long to wait for an idle GPU before failing. Defaults to no wait.')
  parser.add_argument('--gpu_select_poll_interval', type=int, default=None,
                      help='Polling interval in seconds while waiting for an idle GPU.')
  parser.add_argument('--nvidia_smi_path', type=str, default=None,
                      help='Path to nvidia-smi if it is not on PATH.')

  args = parser.parse_args()
  main(args)
