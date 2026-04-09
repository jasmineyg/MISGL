# coding=utf-8

import argparse
import logging
import os

from MISGL.utils import hparam
from MISGL.utils.load_data import GraphDataLoaderWrapper


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
  for name in names:
    if name not in seen:
      seen.add(name)
      deduped.append(name)
  return deduped


def _log_manifest_summary(manifest, split_path):
  build_info = manifest.get('build_info', {}) or {}
  logging.warning('Split file: %s', split_path)
  logging.warning(
    'Protocol: %s | folds=%s | group_stratified=%s',
    manifest.get('protocol'),
    manifest.get('cv_num_folds'),
    build_info.get('used_group_stratified', 'unknown')
  )
  for fold in manifest.get('folds', []):
    logging.warning(
      'Fold %d => samples=%d, labels=%s',
      int(fold.get('fold_id', -1)),
      len(fold.get('sample_indices', [])),
      fold.get('label_hist', {})
    )


def main(args):
  base_hparams = hparam.HParams()
  base_hparams.from_yaml(args.hparam_path)
  if args.processed_data_dir:
    base_hparams.processed_data_dir = args.processed_data_dir

  dataset_names = _parse_data_name_set(getattr(args, 'data_name_set', None))
  if not dataset_names:
    dataset_names = _parse_data_name_set(getattr(base_hparams, 'data_name_set', None))
  if not dataset_names:
    raise ValueError('No dataset specified. Please set data_name_set in the yaml, or pass --data_name_set on the command line.')

  for data_name in dataset_names:
    dataset_hparams = hparam.HParams()
    dataset_hparams.from_yaml(args.hparam_path)
    dataset_hparams.data_name = data_name
    if args.processed_data_dir:
      dataset_hparams.processed_data_dir = args.processed_data_dir

    data_loader = GraphDataLoaderWrapper(dataset_hparams, data_name=data_name)
    split_path = data_loader.get_cv_split_path(ensure_dir=True)

    if os.path.exists(split_path) and not args.overwrite:
      manifest = data_loader.load_cv_split_manifest(split_path)
      logging.warning('Reusing existing split manifest for %s', data_name)
      _log_manifest_summary(manifest, split_path)
      continue

    manifest = data_loader.save_cv_split_manifest(split_path, overwrite=args.overwrite)
    logging.warning('Generated split manifest for %s', data_name)
    _log_manifest_summary(manifest, split_path)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  parser = argparse.ArgumentParser(description='Prepare and persist fixed 10-fold CV split manifests.')
  parser.add_argument('--hparam_path', nargs='?', type=str,
                      default='./config/hparams_testdb.yml',
                      help='The path to the .yml file which contains all the hyperparameters.')
  parser.add_argument('--data_name_set', nargs='*', type=str, default=None,
                      help='Prepare split manifests for one or more datasets.')
  parser.add_argument('--processed_data_dir', nargs='?', type=str, default=None,
                      help='Override processed_data_dir from the yaml for the current run.')
  parser.add_argument('--overwrite', action='store_true',
                      help='Overwrite existing split manifest files.')

  main(parser.parse_args())
