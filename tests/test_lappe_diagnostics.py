import tempfile
import unittest
import os

import numpy as np
import torch

import lappe_diagnostics as diagnostics
from MISGL.models.encoder import MISGLEncoder
from MISGL.utils import hparam
from MISGL.utils import hparams_lib


class LapPEDiagnosticsTest(unittest.TestCase):
    def test_lappe_probe_detects_label_signal(self):
        rng = np.random.default_rng(7)
        labels = np.repeat(np.asarray([0, 1], dtype=np.int64), 50)
        features = np.stack(
            [
                labels + rng.normal(scale=0.15, size=labels.size),
                rng.normal(size=labels.size),
            ],
            axis=1,
        ).astype(np.float32)

        result = diagnostics.probe_space(
            features,
            labels,
            splits=5,
            repeats=1,
            seed=7,
            feature_name='informative',
        )

        self.assertGreater(result['metrics']['auc']['mean'], 0.95)
        self.assertGreater(result['metrics']['acc']['mean'], 0.9)

    def test_topk_diagnostics_exports_selected_lappe(self):
        dataset = diagnostics.synthetic_dataset(seed=11)
        labels = diagnostics.resolve_labels(dataset)
        coarse_adj, _, _ = diagnostics.build_graph_inputs(dataset)
        args = type(
            'Args',
            (),
            {
                'topk_list': '2,4',
                'coarse_topk': 4,
                'lap_pe_dim': 4,
                'spectrum_size': 12,
                'near_zero_tol': 1e-6,
                'pair_sample_size': 1000,
                'seed': 11,
            },
        )()

        result = diagnostics.topk_diagnostics(coarse_adj, labels, args)

        self.assertEqual(result['selected_lap_pe'].shape, (60, 4))
        variants = {row['variant'] for row in result['graph_rows']}
        self.assertEqual(variants, {'unpruned', 'topk_2', 'topk_4'})
        self.assertEqual(len(result['damage_rows']), 2)

    def test_dataset_diagnostic_writes_primary_outputs(self):
        dataset = diagnostics.synthetic_dataset(seed=13)
        args = type(
            'Args',
            (),
            {
                'topk_list': '2,4',
                'coarse_topk': 4,
                'lap_pe_dim': 4,
                'spectrum_size': 12,
                'near_zero_tol': 1e-6,
                'pair_sample_size': 1000,
                'seed': 13,
                'z_mil_path': None,
                'checkpoint': None,
                'hparam_path': None,
                'device': 'cpu',
                'model_batch_size': 16,
                'probe_splits': 3,
                'probe_repeats': 1,
                'knn_k_list': '3,5',
            },
        )()

        with tempfile.TemporaryDirectory() as output_dir:
            summary = diagnostics.diagnose_dataset(
                dataset,
                'synthetic',
                args,
                output_dir,
            )

            self.assertEqual(summary['data_name'], 'synthetic')
            self.assertIn('lap_pe', summary['probes'])
            self.assertEqual(summary['model_utilization']['status'], 'skipped')
            for filename in (
                'summary.json',
                'probe_runs.csv',
                'distance_metrics.csv',
                'knn_consistency.csv',
                'graph_health.csv',
                'eigenvalues.csv',
                'topk_spectral_damage.csv',
            ):
                self.assertTrue(
                    os.path.exists(os.path.join(output_dir, filename)),
                    filename,
                )

    def test_checkpoint_model_utilization_path_runs(self):
        dataset = diagnostics.synthetic_dataset(seed=17)
        labels = diagnostics.resolve_labels(dataset)
        coarse_adj, _, _ = diagnostics.build_graph_inputs(dataset)
        args = type(
            'Args',
            (),
            {
                'lap_pe_dim': 4,
                'spectrum_size': 12,
                'near_zero_tol': 1e-6,
                'device': 'cpu',
                'model_batch_size': 16,
                'seed': 17,
            },
        )()
        _, selected_adj = diagnostics.symmetrize_after_topk(coarse_adj, 4)
        lap_pe = diagnostics.spectrum_and_pe(
            selected_adj,
            lap_pe_dim=4,
            spectrum_size=12,
            near_zero_tol=1e-6,
        )['lap_pe']

        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = os.path.join(temp_dir, 'hparams.yml')
            checkpoint_path = os.path.join(temp_dir, 'checkpoint.pt')
            with open(yaml_path, 'w', encoding='utf-8') as handle:
                handle.write(
                    '\n'.join(
                        [
                            "device: 'cpu'",
                            'channel_list: [2, 4, 4, 4, 1]',
                            'batch_size: 8',
                            'dropout: 0.0',
                            'branch_b:',
                            '  use: false',
                            'use_lappe: true',
                            'lap_pe_dim: 4',
                            'pos_dim: 3',
                            "classifier_input_mode: 'mean_position'",
                        ]
                    )
                )
            hp = hparam.HParams()
            hp.from_yaml(yaml_path)
            hparams_lib.apply_defaults(hp)
            model = MISGLEncoder(hp, data_name='synthetic')
            torch.save({'model_state_dict': model.state_dict()}, checkpoint_path)

            result, artifact = diagnostics.model_utilization_diagnostics(
                dataset,
                labels,
                lap_pe,
                checkpoint_path,
                yaml_path,
                args,
                'synthetic',
            )

            self.assertEqual(result['status'], 'ok')
            self.assertIn('shuffled_lappe_metric_drop', result)
            self.assertEqual(artifact['features'].shape[0], labels.shape[0])
            self.assertEqual(artifact['pos_emb'].shape[0], labels.shape[0])


if __name__ == '__main__':
    unittest.main()
