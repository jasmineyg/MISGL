# coding=utf-8

"""Utilities for persisting experiment summaries to Excel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from contextlib import contextmanager

import pandas as pd

if os.name == 'nt':
  import msvcrt
else:
  import fcntl


TEST_SHEET_NAME = 'test'
TRAIN_VAL_SHEET_NAME = 'train_val'
LOCK_TIMEOUT_SECONDS = 120
LOCK_POLL_SECONDS = 0.1
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
  sheet_df = sheet_df.loc[:, columns]
  for col in columns:
    sheet_df[col] = sheet_df[col].astype(object)
  return sheet_df


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


def _is_blank(value):
  return pd.isna(value) or str(value).strip() == ''


def _safe_metric(results, dataset_name, split_name, metric_name):
  dataset_result = results.get(dataset_name, {}) or {}
  split_result = dataset_result.get(split_name, {}) or {}
  value = split_result.get(metric_name, '')
  if value is None:
    return ''
  return str(value)


def _test_updates(results):
  updates = {}
  for dataset_name in DATASET_ORDER:
    acc = _safe_metric(results, dataset_name, 'test', 'acc')
    f1 = _safe_metric(results, dataset_name, 'test', 'f1')
    if not _is_blank(acc):
      updates['{}_ACC'.format(dataset_name)] = acc
    if not _is_blank(f1):
      updates['{}_F1'.format(dataset_name)] = f1
  return updates


def _build_test_row(experiment_name, results):
  row = {'timestamp': experiment_name}
  row.update(_test_updates(results))
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


def _candidate_name(base_name, suffix):
  if suffix == 0:
    return base_name
  return '{}({})'.format(base_name, suffix)


def _find_test_row(test_df, train_val_df, experiment_name, updates):
  base_name = str(experiment_name).strip()
  if not base_name:
    raise ValueError('experiment_name must not be empty')

  existing_names = _normalize_existing_names(test_df['timestamp'].tolist())
  existing_names.update(_normalize_existing_names(train_val_df['timestamp'].tolist()))
  if not updates:
    return _make_unique_experiment_name(base_name, existing_names), None

  normalized_timestamps = test_df['timestamp'].apply(
    lambda value: '' if pd.isna(value) else str(value).strip()
  )
  suffix = 0
  while True:
    candidate = _candidate_name(base_name, suffix)
    matching_indices = test_df.index[normalized_timestamps == candidate].tolist()
    for row_index in matching_indices:
      if all(_is_blank(test_df.at[row_index, column]) for column in updates):
        return candidate, row_index
    if candidate not in existing_names:
      return candidate, None
    suffix += 1


def _lock_file(lock_file):
  if os.name == 'nt':
    lock_file.seek(0)
    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
  else:
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(lock_file):
  if os.name == 'nt':
    lock_file.seek(0)
    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
  else:
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _workbook_lock(excel_path):
  lock_path = '{}.lock'.format(excel_path)
  deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
  with open(lock_path, 'a+b') as lock_file:
    if os.path.getsize(lock_path) == 0:
      lock_file.write(b'\0')
      lock_file.flush()

    while True:
      try:
        _lock_file(lock_file)
        break
      except (IOError, OSError):
        if time.monotonic() >= deadline:
          raise TimeoutError(
            'Timed out waiting for experiment results lock: {}'.format(lock_path)
          )
        time.sleep(LOCK_POLL_SECONDS)

    try:
      yield
    finally:
      _unlock_file(lock_file)


def save_experiment_results(excel_path, experiment_name, results):
  """Add one experiment summary to an Excel workbook.

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

  with _workbook_lock(excel_path):
    test_df = _read_sheet(excel_path, TEST_SHEET_NAME, TEST_COLUMNS)
    train_val_df = _read_sheet(excel_path, TRAIN_VAL_SHEET_NAME, TRAIN_VAL_COLUMNS)
    updates = _test_updates(results)
    saved_name, row_index = _find_test_row(
      test_df,
      train_val_df,
      experiment_name,
      updates,
    )

    if row_index is None:
      test_df = pd.concat(
        [test_df, pd.DataFrame([_build_test_row(saved_name, results)], columns=TEST_COLUMNS)],
        ignore_index=True,
      )
    else:
      for column, value in updates.items():
        test_df.at[row_index, column] = value

    train_val_rows = _build_train_val_rows(saved_name, results)
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

  return saved_name
