import os
import tempfile
import unittest

import torch

from MISGL.bin.train_eval import _fold_checkpoint_path
from MISGL.bin.train_eval import _save_fold_checkpoint
from MISGL.utils import hparam
from MISGL.utils import hparams_lib


class FoldCheckpointTest(unittest.TestCase):
    def test_saves_best_fold_checkpoint_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            params = hparam.HParams(
                model_save_path=temp_dir,
                timestamp='ogbn_arxiv_pos_lappe',
                fold_checkpoint_dir=os.path.join(temp_dir, 'folds'),
                lap_pe_tensor=torch.ones(2, 2),
            )
            hparams_lib.apply_defaults(params)
            model = torch.nn.Linear(3, 1)
            split = {
                'train_indices': [0, 1],
                'val_indices': [2],
                'test_indices': [3],
            }
            best_val = {'epoch': 7, 'acc': 0.8, 'loss': 0.4, 'train_loss': 0.3}

            path = _save_fold_checkpoint(
                model,
                params,
                data_name='ogbn_arxiv',
                fold_idx=2,
                seed=1026,
                split_meta=split,
                best_val_result=best_val,
            )

            self.assertEqual(path, _fold_checkpoint_path(params, 2))
            self.assertTrue(os.path.exists(path))
            payload = torch.load(path, map_location='cpu')
            self.assertEqual(payload['fold_idx'], 2)
            self.assertEqual(payload['seed'], 1026)
            self.assertEqual(payload['best_val']['epoch'], 7)
            self.assertNotIn('lap_pe_tensor', payload['hparams'])
            self.assertEqual(set(payload['model_state_dict']), set(model.state_dict()))


if __name__ == '__main__':
    unittest.main()
