import multiprocessing
import os
import tempfile
import unittest

import pandas as pd

from MISGL.utils.experiment_results import save_experiment_results


def _results_for(dataset_name, acc, f1):
  return {
    dataset_name: {
      'test': {
        'acc': acc,
        'f1': f1,
      },
    },
  }


def _save_in_process(excel_path, dataset_name, acc, f1, start_event, output_queue):
  start_event.wait()
  try:
    saved_name = save_experiment_results(
      excel_path,
      'loss',
      _results_for(dataset_name, acc, f1),
    )
    output_queue.put(('ok', saved_name))
  except Exception as exc:
    output_queue.put(('error', repr(exc)))


class ExperimentResultsTest(unittest.TestCase):

  def test_different_datasets_reuse_the_same_timestamp_row(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      excel_path = os.path.join(temp_dir, 'results.xlsx')

      first_name = save_experiment_results(
        excel_path,
        'loss',
        _results_for('ogbn_arxiv', '0.8645', '0.8513'),
      )
      second_name = save_experiment_results(
        excel_path,
        'loss',
        _results_for('reddit', '0.8678', '0.8046'),
      )

      test_df = pd.read_excel(excel_path, sheet_name='test', engine='openpyxl')
      self.assertEqual(first_name, 'loss')
      self.assertEqual(second_name, 'loss')
      self.assertEqual(test_df['timestamp'].tolist(), ['loss'])
      self.assertEqual(str(test_df.loc[0, 'ogbn_arxiv_ACC']), '0.8645')
      self.assertEqual(str(test_df.loc[0, 'reddit_ACC']), '0.8678')

  def test_repeated_datasets_fill_matching_suffix_rows(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      excel_path = os.path.join(temp_dir, 'results.xlsx')

      saved_names = [
        save_experiment_results(
          excel_path,
          'loss',
          _results_for('ogbn_arxiv', 'arxiv-1', 'arxiv-f1-1'),
        ),
        save_experiment_results(
          excel_path,
          'loss',
          _results_for('ogbn_arxiv', 'arxiv-2', 'arxiv-f1-2'),
        ),
        save_experiment_results(
          excel_path,
          'loss',
          _results_for('reddit', 'reddit-1', 'reddit-f1-1'),
        ),
        save_experiment_results(
          excel_path,
          'loss',
          _results_for('reddit', 'reddit-2', 'reddit-f1-2'),
        ),
      ]

      test_df = pd.read_excel(excel_path, sheet_name='test', engine='openpyxl')
      self.assertEqual(saved_names, ['loss', 'loss(1)', 'loss', 'loss(1)'])
      self.assertEqual(test_df['timestamp'].tolist(), ['loss', 'loss(1)'])
      self.assertEqual(test_df['reddit_ACC'].tolist(), ['reddit-1', 'reddit-2'])

  def test_parallel_datasets_share_the_same_timestamp_row(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      excel_path = os.path.join(temp_dir, 'results.xlsx')
      context = multiprocessing.get_context('spawn')
      start_event = context.Event()
      output_queue = context.Queue()
      processes = [
        context.Process(
          target=_save_in_process,
          args=(excel_path, 'ogbn_arxiv', 'arxiv-acc', 'arxiv-f1', start_event, output_queue),
        ),
        context.Process(
          target=_save_in_process,
          args=(excel_path, 'reddit', 'reddit-acc', 'reddit-f1', start_event, output_queue),
        ),
      ]

      for process in processes:
        process.start()
      start_event.set()
      for process in processes:
        process.join(timeout=15)

      self.assertTrue(all(not process.is_alive() for process in processes))
      outcomes = [output_queue.get(timeout=2) for _ in processes]
      self.assertEqual(outcomes.count(('ok', 'loss')), 2, outcomes)

      test_df = pd.read_excel(excel_path, sheet_name='test', engine='openpyxl')
      self.assertEqual(test_df['timestamp'].tolist(), ['loss'])
      self.assertEqual(test_df.loc[0, 'ogbn_arxiv_ACC'], 'arxiv-acc')
      self.assertEqual(test_df.loc[0, 'reddit_ACC'], 'reddit-acc')


if __name__ == '__main__':
  unittest.main()
