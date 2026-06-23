import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from MISGL.bin import train_eval
from MISGL.utils import get_loss
from MISGL.utils.evaluate import evaluate
from MISGL.utils.global_variables import g_key


def _hparams(**overrides):
    values = {
        'loss_type': 'bce',
        'focal_gamma': 2.0,
        'loss_pos_weight': None,
        'label_smoothing': 0.0,
        'device': 'cpu',
        'branch_b': {'use': False},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class LossAblationTest(unittest.TestCase):
    def test_focal_gamma_zero_equals_bce(self):
        logits = torch.tensor([-2.0, -0.2, 0.4, 2.5])
        targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
        bce = get_loss.binary_classification_loss(logits, targets, _hparams())
        focal = get_loss.binary_classification_loss(
            logits, targets, _hparams(loss_type='focal', focal_gamma=0.0)
        )
        torch.testing.assert_close(focal, bce)

    def test_focal_downweights_easy_example_more(self):
        logits = torch.tensor([5.0, 0.1])
        targets = torch.ones(2)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weights = torch.pow(-torch.expm1(-bce), 2.0)
        self.assertLess(float(weights[0]), float(weights[1]))

    def test_weighted_bce_matches_pytorch(self):
        logits = torch.tensor([-1.2, 0.7, 1.4])
        targets = torch.tensor([0.0, 1.0, 1.0])
        actual = get_loss.binary_classification_loss(
            logits,
            targets,
            _hparams(loss_type='weighted_bce', loss_pos_weight=1.75),
        )
        expected = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=torch.tensor(1.75)
        )
        torch.testing.assert_close(actual, expected)

    def test_label_smoothing_and_attention_shape_apply_to_focal(self):
        logits = torch.tensor([0.3, -0.5])
        targets = torch.tensor([1.0, 0.0])
        attention = torch.tensor([0.8, 0.2, 0.5, 0.5])
        batch = torch.tensor([0, 0, 1, 1])
        hparams = _hparams(
            loss_type='focal',
            label_smoothing=0.1,
            branch_b={
                'use': True,
                'attention_shape_loss_enabled': True,
                'attention_shape_loss_weight': 0.2,
            },
        )
        model_output = {
            'ypred_A': logits,
            'branch_b': {'a': attention, 'batch': batch},
        }
        expected = get_loss.binary_classification_loss(logits, targets, hparams)
        expected = expected + 0.2 * get_loss.mil_attention_shape_loss(
            attention, batch, targets
        )
        actual = get_loss.fused_loss(model_output, targets, 3, hparams)
        torch.testing.assert_close(actual, expected)

    def test_invalid_loss_parameters_raise(self):
        logits = torch.zeros(2)
        targets = torch.tensor([0.0, 1.0])
        with self.assertRaises(ValueError):
            get_loss.binary_classification_loss(
                logits, targets, _hparams(loss_type='unknown')
            )
        with self.assertRaises(ValueError):
            get_loss.binary_classification_loss(
                logits, targets, _hparams(loss_type='focal', focal_gamma=-1.0)
            )
        with self.assertRaises(ValueError):
            get_loss.binary_classification_loss(
                logits, targets, _hparams(loss_type='weighted_bce')
            )

    def test_training_fold_requires_both_classes(self):
        dataset = SimpleNamespace(
            processed_graph_list=[{g_key.y: torch.tensor(0)}]
        )
        with self.assertRaisesRegex(ValueError, 'no positive'):
            train_eval._training_label_stats(SimpleNamespace(dataset=dataset))


class _ConstantModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer('logits', torch.tensor(logits, dtype=torch.float32))

    def forward(self, batch):
        return self.logits[: batch[g_key.y].numel()]


class EvaluateTest(unittest.TestCase):
    def test_single_class_auc_is_none(self):
        dataset = [{g_key.y: torch.tensor([0, 0], dtype=torch.long)}]
        result = evaluate(dataset, _ConstantModel([-1.0, -2.0]), _hparams())
        self.assertIsNone(result['roc_auc'])
        self.assertIsNone(result['pr_auc'])
        self.assertEqual(result['tn'], 2)
        self.assertEqual(result['tp'], 0)

    def test_full_metric_set(self):
        dataset = [{g_key.y: torch.tensor([0, 1, 1, 0], dtype=torch.long)}]
        result = evaluate(
            dataset,
            _ConstantModel([-2.0, 2.0, -0.5, 0.5]),
            _hparams(),
        )
        for key in (
            'acc', 'F1', 'prec', 'rec', 'balanced_acc', 'roc_auc', 'pr_auc',
            'tn', 'fp', 'fn', 'tp',
        ):
            self.assertIn(key, result)
        self.assertTrue(np.isfinite(result['roc_auc']))


if __name__ == '__main__':
    unittest.main()
