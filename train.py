# coding=utf-8

# function main(args) and top-level imports
import os
import torch
import numpy as np
import logging
import argparse
import random
import time

from gnn_hpool.utils import hparam
from gnn_hpool.utils import reproducibility

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


def main(args):

  # reproducibility
  reproducibility.set_seed(1024)

  from gnn_hpool.bin import train_eval

  base_hparams = hparam.HParams()
  base_hparams.from_yaml(args.hparam_path)
  if args.processed_data_dir:
    base_hparams.processed_data_dir = args.processed_data_dir

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
    hparams.data_name = data_name
    if args.processed_data_dir:
      hparams.processed_data_dir = args.processed_data_dir

    if hparams.device == 'cuda':
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

    os.environ['CUDA_VISIBLE_DEVICES'] = hparams.cuda_visible_devices
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

  args = parser.parse_args()
  main(args)
