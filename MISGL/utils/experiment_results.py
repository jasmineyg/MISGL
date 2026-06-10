# coding=utf-8

"""Utilities for persisting experiment summaries to Excel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import pandas as pd


TEST_SHEET_NAME = 'test'
TRAIN_VAL_SHEET_NAME = 'train_val'
DATASET_ORDER = ('ogbn_arxiv', 'ogbn_products', 'reddit')
TRAIN_VAL_SPLITS = ('train', 'val')
TEST_COLUMNS = [
  'timestamp',
  'ogbn_arxiv_ACC',
  'ogbn_arxiv_F1',
  'ogbn_products_ACC',
  'ogbn_products_F1',
  'reddit_ACC',
  'reddit_F1',
]
TRAIN_VAL_COLUMNS = ['timestamp', 'dataset', 'split', 'ACC', 'F1']


def _empty_test_df():
  return pd.DataFrame(columns=TEST_COLUMNS)


def _empty_train_val_df():
  return pd.DataFrame(columns=TRAIN_VAL_COLUMNS)


def _read_sheet(excel_path, sheet_name, columns):
  if not os.path.exists(excel_path):
    return pd.DataFrame(columns=columns)
  try:
    sheet_df = pd.read_excel(excel_path, sheet_name=sheet_name, engine='openpyxl')
  except ValueError:
    return pd.DataFrame(columns=columns)

  for col in columns:
    if col not in sheet_df.columns:
      sheet_df[col] = ''
  return sheet_df.loc[:, columns]


def _normalize_existing_names(values):
  names = set()
  for value in values:
    if pd.isna(value):
      continue
    name = str(value).strip()
    if name:
      names.add(name)
  return names


def _make_unique_experiment_name(experiment_name, existing_names):
  base_name = str(experiment_name).strip()
  if not base_name:
    raise ValueError('experiment_name must not be empty')
  if base_name not in existing_names:
    return base_name

  suffix = 1
  while True:
    candidate = '{}({})'.format(base_name, suffix)
    if candidate not in existing_names:
      return candidate
    suffix += 1


def _safe_metric(results, dataset_name, split_name, metric_name):
  dataset_result = results.get(dataset_name, {}) or {}
  split_result = dataset_result.get(split_name, {}) or {}
  value = split_result.get(metric_name, '')
  if value is None:
    return ''
  return str(value)


def _build_test_row(experiment_name, results):
  row = {'timestamp': experiment_name}
  for dataset_name in DATASET_ORDER:
    row['{}_ACC'.format(dataset_name)] = _safe_metric(results, dataset_name, 'test', 'acc')
    row['{}_F1'.format(dataset_name)] = _safe_metric(results, dataset_name, 'test', 'f1')
  return row


def _build_train_val_rows(experiment_name, results):
  rows = []
  for dataset_name in DATASET_ORDER:
    dataset_result = results.get(dataset_name, {}) or {}
    for split_name in TRAIN_VAL_SPLITS:
      if split_name not in dataset_result:
        continue
      rows.append({
        'timestamp': experiment_name,
        'dataset': dataset_name,
        'split': split_name,
        'ACC': _safe_metric(results, dataset_name, split_name, 'acc'),
        'F1': _safe_metric(results, dataset_name, split_name, 'f1'),
      })
  return rows


def save_experiment_results(excel_path, experiment_name, results):
  """Append one experiment summary to an Excel workbook.

  Args:
    excel_path: Full path to the target Excel file.
    experiment_name: Base experiment name from config, e.g. MISGL-LapPE.
    results: Nested metrics dictionary:
      {
        "ogbn_arxiv": {
          "train": {"acc": "...", "f1": "..."},
          "val": {"acc": "...", "f1": "..."},
          "test": {"acc": "...", "f1": "..."},
        },
        ...
      }

  Returns:
    The unique experiment name actually written to Excel.
  """
  if not excel_path:
    raise ValueError('excel_path must not be empty')
  if not isinstance(results, dict):
    raise ValueError('results must be a dict')

  excel_path = os.path.abspath(excel_path)
  parent_dir = os.path.dirname(excel_path)
  if parent_dir:
    os.makedirs(parent_dir, exist_ok=True)

  test_df = _read_sheet(excel_path, TEST_SHEET_NAME, TEST_COLUMNS)
  train_val_df = _read_sheet(excel_path, TRAIN_VAL_SHEET_NAME, TRAIN_VAL_COLUMNS)
  existing_names = _normalize_existing_names(test_df['timestamp'].tolist())
  existing_names.update(_normalize_existing_names(train_val_df['timestamp'].tolist()))
  unique_name = _make_unique_experiment_name(experiment_name, existing_names)

  test_df = pd.concat(
    [test_df, pd.DataFrame([_build_test_row(unique_name, results)], columns=TEST_COLUMNS)],
    ignore_index=True,
  )
  train_val_rows = _build_train_val_rows(unique_name, results)
  train_val_df = pd.concat(
    [train_val_df, pd.DataFrame(train_val_rows, columns=TRAIN_VAL_COLUMNS)],
    ignore_index=True,
  )

  writer_mode = 'a' if os.path.exists(excel_path) else 'w'
  with pd.ExcelWriter(
    excel_path,
    engine='openpyxl',
    mode=writer_mode,
    if_sheet_exists='replace' if writer_mode == 'a' else None,
  ) as writer:
    test_df.to_excel(writer, sheet_name=TEST_SHEET_NAME, index=False)
    train_val_df.to_excel(writer, sheet_name=TRAIN_VAL_SHEET_NAME, index=False)

  return unique_name
