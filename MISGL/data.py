"""Strict graph loading, sparse batching, and grouped cross-validation."""

import pickle
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset

from MISGL.keys import BAG_INDEX, EDGE_INDEX, LABEL, SAMPLE_INDEX, STRUCTURE, X


def _binary_labels(values: Any, expected_size: int) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1 or labels.shape[0] != expected_size:
        raise ValueError(
            'subgraph_labels must be a one-dimensional array with one label '
            'per subgraph_structure.'
        )
    if labels.dtype.kind not in 'biuf':
        raise TypeError('subgraph_labels must contain numeric binary values.')
    if not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError('subgraph_labels must contain only 0 and 1.')
    return labels.astype(np.int64, copy=False)


def _index_array(values: Any, name: str) -> np.ndarray:
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise TypeError('{} must be a one-dimensional integer sequence.'.format(name))
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError('{} must be one-dimensional.'.format(name))
    if array.size == 0:
        return np.empty(0, dtype=np.int64)
    if array.dtype.kind not in 'iu':
        raise TypeError('{} must contain integers.'.format(name))
    return array.astype(np.int64, copy=False)


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError('{} must be an integer.'.format(name))
    result = int(value)
    if result < minimum:
        raise ValueError('{} must be at least {}.'.format(name, minimum))
    return result


class GraphTensorDataset(Dataset):
    """Preprocess each active NetworkX graph once into sparse CPU tensors."""

    def __init__(
        self,
        subgraphs: Sequence[nx.Graph],
        labels: Any,
        sample_indices: Any,
        include_structure: bool,
    ) -> None:
        if not isinstance(subgraphs, (list, tuple)) or not subgraphs:
            raise ValueError('subgraphs must be a non-empty list or tuple.')
        if type(include_structure) is not bool:
            raise TypeError('include_structure must be a bool.')

        label_array = _binary_labels(labels, len(subgraphs))
        index_array = _index_array(sample_indices, 'sample_indices')
        if index_array.shape[0] != len(subgraphs):
            raise ValueError('sample_indices must contain one index per subgraph.')

        self._samples: List[Dict[str, torch.Tensor]] = []
        feature_dim = None

        for graph_position, graph in enumerate(subgraphs):
            sample_index = int(index_array[graph_position])
            if not isinstance(graph, nx.Graph):
                raise TypeError(
                    'subgraph_structures[{}] must be a NetworkX graph.'.format(
                        sample_index
                    )
                )

            node_ids = list(graph.nodes())
            if not node_ids:
                raise ValueError(
                    'subgraph_structures[{}] has no nodes.'.format(sample_index)
                )

            node_features = []
            for node_id in node_ids:
                if 'features' not in graph.nodes[node_id]:
                    raise ValueError(
                        "Node {!r} in subgraph_structures[{}] is missing 'features'.".format(
                            node_id, sample_index
                        )
                    )
                features = np.asarray(graph.nodes[node_id]['features'], dtype=np.float32)
                if features.ndim != 1 or features.size == 0:
                    raise ValueError(
                        "Node {!r} in subgraph_structures[{}] must have a non-empty "
                        "one-dimensional 'features' array.".format(node_id, sample_index)
                    )
                if not np.isfinite(features).all():
                    raise ValueError(
                        "Node {!r} in subgraph_structures[{}] has non-finite features.".format(
                            node_id, sample_index
                        )
                    )
                if feature_dim is None:
                    feature_dim = int(features.size)
                elif features.size != feature_dim:
                    raise ValueError(
                        'All node feature vectors must have the same dimension; '
                        'expected {}, got {} in subgraph_structures[{}].'.format(
                            feature_dim, features.size, sample_index
                        )
                    )
                node_features.append(features)

            x = np.stack(node_features).astype(np.float32, copy=False)
            adjacency = sp.csr_matrix(
                nx.to_scipy_sparse_array(
                    graph,
                    nodelist=node_ids,
                    dtype=np.float32,
                    weight='weight',
                    format='csr',
                )
            )
            if not np.isfinite(adjacency.data).all():
                raise ValueError(
                    'subgraph_structures[{}] has non-finite edge weights.'.format(
                        sample_index
                    )
                )
            adjacency.eliminate_zeros()

            topology = adjacency.copy()
            topology.data = np.ones(topology.nnz, dtype=np.float32)
            topology = topology.maximum(
                sp.eye(len(node_ids), dtype=np.float32, format='csr')
            ).tocoo()
            edge_index = np.vstack((topology.row, topology.col)).astype(
                np.int64, copy=False
            )

            sample = {
                X: torch.from_numpy(x),
                EDGE_INDEX: torch.from_numpy(edge_index),
                LABEL: torch.tensor(label_array[graph_position], dtype=torch.long),
                SAMPLE_INDEX: torch.tensor(sample_index, dtype=torch.long),
            }
            if include_structure:
                structure = self._compute_structural_features_np(
                    adjacency.toarray()
                )
                sample[STRUCTURE] = torch.from_numpy(structure)
            self._samples.append(sample)

        self.feature_dim = int(feature_dim)

    @staticmethod
    def _compute_structural_features_np(adjacency: np.ndarray) -> np.ndarray:
        adjacency_bool = adjacency != 0
        adjacency_bool = np.logical_or(adjacency_bool, adjacency_bool.T)
        np.fill_diagonal(adjacency_bool, False)
        adjacency_float = adjacency_bool.astype(np.float32, copy=False)

        node_count = adjacency_float.shape[0]
        max_degree = max(float(node_count - 1), 1.0)
        degree = adjacency_float.sum(axis=-1)
        degree_normalized = degree / max_degree
        log_degree_normalized = np.log1p(degree) / np.log1p(max_degree)

        neighbor_degree_sum = adjacency_float @ degree
        average_neighbor_degree = neighbor_degree_sum / np.maximum(degree, 1.0)
        average_neighbor_degree_normalized = average_neighbor_degree / max_degree

        two_hop_walk_count = neighbor_degree_sum
        two_hop_denominator = max(max_degree ** 2, 1.0)
        two_hop_walk_log_normalized = np.log1p(two_hop_walk_count) / np.log1p(
            two_hop_denominator
        )

        two_path_count = adjacency_float @ adjacency_float
        closed_wedge_count = (two_path_count * adjacency_float).sum(axis=-1)
        triangle_count = closed_wedge_count / 2.0
        max_triangle_count = max(
            max_degree * (max_degree - 1.0) / 2.0, 1.0
        )
        triangle_count_log_normalized = np.log1p(triangle_count) / np.log1p(
            max_triangle_count
        )

        possible_wedge_count = degree * (degree - 1.0)
        clustering_coefficient = np.divide(
            closed_wedge_count,
            np.maximum(possible_wedge_count, 1.0),
            out=np.zeros_like(degree),
            where=possible_wedge_count > 0,
        )

        core_number_normalized = GraphTensorDataset._compute_core_number_norm_np(
            adjacency_float, max_degree
        )
        return np.stack(
            [
                degree_normalized,
                log_degree_normalized,
                average_neighbor_degree_normalized,
                two_hop_walk_log_normalized,
                triangle_count_log_normalized,
                clustering_coefficient,
                core_number_normalized,
            ],
            axis=-1,
        ).astype(np.float32, copy=False)

    @staticmethod
    def _compute_core_number_norm_np(
        adjacency: np.ndarray,
        max_degree: float,
    ) -> np.ndarray:
        node_count = adjacency.shape[0]
        remaining = np.ones(node_count, dtype=bool)
        working_degree = adjacency.sum(axis=-1).astype(np.float32, copy=True)
        local_core = np.zeros(node_count, dtype=np.float32)
        running_core = 0.0

        for _ in range(node_count):
            masked_degree = np.where(remaining, working_degree, np.inf)
            node_index = int(np.argmin(masked_degree))
            running_core = max(running_core, float(masked_degree[node_index]))
            local_core[node_index] = running_core
            remaining[node_index] = False
            working_degree = np.maximum(
                working_degree - adjacency[:, node_index], 0.0
            )

        return local_core / max_degree

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self._samples[index]


def collate_graph_batch(
    samples: Sequence[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Concatenate graphs into one disconnected sparse graph batch."""
    if not samples:
        raise ValueError('Cannot collate an empty graph batch.')

    include_structure = STRUCTURE in samples[0]
    if any((STRUCTURE in sample) != include_structure for sample in samples):
        raise ValueError('All samples in a batch must use the same structure schema.')

    node_features = []
    edge_indices = []
    bag_indices = []
    node_offset = 0
    for bag_index, sample in enumerate(samples):
        node_count = sample[X].shape[0]
        node_features.append(sample[X])
        edge_indices.append(sample[EDGE_INDEX] + node_offset)
        bag_indices.append(torch.full((node_count,), bag_index, dtype=torch.long))
        node_offset += node_count

    batch = {
        X: torch.cat(node_features, dim=0),
        EDGE_INDEX: torch.cat(edge_indices, dim=1),
        BAG_INDEX: torch.cat(bag_indices, dim=0),
        LABEL: torch.stack([sample[LABEL] for sample in samples]),
        SAMPLE_INDEX: torch.stack([sample[SAMPLE_INDEX] for sample in samples]),
    }
    if include_structure:
        batch[STRUCTURE] = torch.cat(
            [sample[STRUCTURE] for sample in samples], dim=0
        )
    return batch


@dataclass
class DatasetBundle:
    """Loaded graph tensors and one in-memory set of grouped stratified folds."""

    feature_dim: int
    labels: np.ndarray
    sample_indices: np.ndarray
    subgraph_ids: np.ndarray
    dataset: GraphTensorDataset
    original_graph: Any
    assignment_matrix: Any
    _fold_indices: Tuple[np.ndarray, ...] = field(repr=False)
    _seed: int = field(repr=False)

    @classmethod
    def load(
        cls,
        config: Any,
        dataset_name: str,
        require_position_graph: bool,
    ) -> 'DatasetBundle':
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError('dataset_name must be a non-empty string.')
        if not isinstance(require_position_graph, bool):
            raise TypeError('require_position_graph must be a bool.')
        if type(config.mil_head.enabled) is not bool:
            raise TypeError('config.mil_head.enabled must be a bool.')

        dataset_path = Path(config.data_dir) / '{}_processed.pkl'.format(dataset_name)
        with dataset_path.open('rb') as handle:
            raw = pickle.load(handle)

        if not isinstance(raw, Mapping):
            raise TypeError('The processed dataset root must be a mapping.')
        required_keys = ('subgraph_structures', 'train_test_split', 'subgraph_labels')
        missing_keys = [key for key in required_keys if key not in raw]
        if missing_keys:
            raise KeyError('Processed dataset is missing keys: {}.'.format(', '.join(missing_keys)))

        subgraphs = raw['subgraph_structures']
        if not isinstance(subgraphs, (list, tuple)) or not subgraphs:
            raise ValueError('subgraph_structures must be a non-empty list or tuple.')

        split_schema = raw['train_test_split']
        if not isinstance(split_schema, Mapping):
            raise TypeError('train_test_split must be a mapping.')
        split_keys = ('train_indices', 'test_indices')
        missing_split_keys = [key for key in split_keys if key not in split_schema]
        if missing_split_keys:
            raise KeyError(
                'train_test_split is missing keys: {}.'.format(', '.join(missing_split_keys))
            )

        train_ids = _index_array(split_schema['train_indices'], 'train_indices')
        test_ids = _index_array(split_schema['test_indices'], 'test_indices')
        sample_indices = np.concatenate((train_ids, test_ids))
        if sample_indices.size == 0:
            raise ValueError('train_indices and test_indices cannot both be empty.')
        if np.unique(sample_indices).size != sample_indices.size:
            raise ValueError('train_indices and test_indices must not contain duplicate indices.')
        if sample_indices.min() < 0 or sample_indices.max() >= len(subgraphs):
            raise IndexError(
                'train_indices/test_indices contain values outside subgraph_structures.'
            )

        all_labels = _binary_labels(raw['subgraph_labels'], len(subgraphs))
        labels = all_labels[sample_indices]
        active_subgraphs = [subgraphs[int(index)] for index in sample_indices]
        subgraph_ids = cls._read_subgraph_ids(active_subgraphs, sample_indices)
        group_ids = cls._read_group_ids(raw, len(subgraphs), sample_indices)

        folds = _integer(config.folds, 'config.folds', 3)
        seed = _integer(config.seed, 'config.seed', 0)
        fold_indices = cls._build_grouped_folds(labels, group_ids, folds, seed)

        original_graph = raw.get('original_graph')
        assignment_matrix = raw.get('assignment_matrix')
        if require_position_graph:
            missing_position_keys = [
                key for key in ('original_graph', 'assignment_matrix')
                if key not in raw or raw[key] is None
            ]
            if missing_position_keys:
                raise KeyError(
                    'Position graph requires dataset keys: {}.'.format(
                        ', '.join(missing_position_keys)
                    )
                )
            cls._validate_position_schema(
                original_graph,
                assignment_matrix,
                subgraph_ids,
            )

        tensor_dataset = GraphTensorDataset(
            active_subgraphs,
            labels,
            sample_indices,
            include_structure=config.mil_head.enabled,
        )
        return cls(
            feature_dim=tensor_dataset.feature_dim,
            labels=labels.copy(),
            sample_indices=sample_indices.copy(),
            subgraph_ids=subgraph_ids,
            dataset=tensor_dataset,
            original_graph=original_graph,
            assignment_matrix=assignment_matrix,
            _fold_indices=fold_indices,
            _seed=seed,
        )

    @staticmethod
    def _read_subgraph_ids(
        subgraphs: Sequence[nx.Graph],
        sample_indices: np.ndarray,
    ) -> np.ndarray:
        subgraph_ids = []
        for graph, sample_index in zip(subgraphs, sample_indices):
            if not isinstance(graph, nx.Graph):
                raise TypeError(
                    'subgraph_structures[{}] must be a NetworkX graph.'.format(
                        int(sample_index)
                    )
                )
            if 'subgraph_id' not in graph.graph:
                raise KeyError(
                    "subgraph_structures[{}] is missing graph.graph['subgraph_id'].".format(
                        int(sample_index)
                    )
                )
            subgraph_id = graph.graph['subgraph_id']
            if isinstance(subgraph_id, bool) or not isinstance(subgraph_id, Integral):
                raise TypeError(
                    "subgraph_structures[{}] graph.graph['subgraph_id'] must be an integer.".format(
                        int(sample_index)
                    )
                )
            subgraph_ids.append(int(subgraph_id))

        result = np.asarray(subgraph_ids, dtype=np.int64)
        if np.unique(result).size != result.size:
            raise ValueError("Active graph.graph['subgraph_id'] values must be unique.")
        if result.min() < 0:
            raise ValueError("graph.graph['subgraph_id'] values must be non-negative.")
        return result

    @staticmethod
    def _read_group_ids(
        raw: Mapping[str, Any],
        subgraph_count: int,
        sample_indices: np.ndarray,
    ) -> np.ndarray:
        if 'group_ids' not in raw:
            return sample_indices.copy()

        values = raw['group_ids']
        if not isinstance(values, (list, tuple, np.ndarray)):
            raise TypeError('group_ids must be a one-dimensional sequence.')
        array = np.asarray(values)
        if array.ndim != 1 or array.shape[0] != subgraph_count:
            raise ValueError(
                'group_ids must contain one value per subgraph_structure.'
            )

        normalized = []
        id_kind = None
        for value in array:
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, bool):
                raise TypeError('group_ids values must be integers or non-empty strings.')
            if isinstance(value, Integral):
                current_kind = int
                value = int(value)
            elif isinstance(value, str) and value:
                current_kind = str
            else:
                raise TypeError('group_ids values must be integers or non-empty strings.')
            if id_kind is None:
                id_kind = current_kind
            elif current_kind is not id_kind:
                raise TypeError('group_ids must not mix integer and string values.')
            normalized.append(value)

        dtype = np.int64 if id_kind is int else object
        return np.asarray(normalized, dtype=dtype)[sample_indices]

    @staticmethod
    def _build_grouped_folds(
        labels: np.ndarray,
        group_ids: np.ndarray,
        fold_count: int,
        seed: int,
    ) -> Tuple[np.ndarray, ...]:
        unique_groups, group_inverse = np.unique(group_ids, return_inverse=True)
        group_sizes = np.bincount(group_inverse)
        positive_counts = np.bincount(group_inverse, weights=labels)
        group_labels = np.rint(positive_counts / group_sizes).astype(np.int64)

        class_counts = np.bincount(group_labels, minlength=2)
        if np.count_nonzero(class_counts) != 2:
            raise ValueError('Active groups must contain both binary label classes.')
        if class_counts.min() < fold_count:
            raise ValueError(
                'Each group-label class needs at least config.folds={} groups; '
                'class counts are {}.'.format(fold_count, class_counts.tolist())
            )

        splitter = StratifiedKFold(
            n_splits=fold_count,
            shuffle=True,
            random_state=seed,
        )
        fold_indices = []
        for _, test_group_positions in splitter.split(unique_groups, group_labels):
            selected_groups = np.zeros(unique_groups.shape[0], dtype=bool)
            selected_groups[test_group_positions] = True
            sample_positions = np.flatnonzero(selected_groups[group_inverse])
            fold_indices.append(sample_positions.astype(np.int64, copy=False))
        return tuple(fold_indices)

    @staticmethod
    def _validate_position_schema(
        original_graph: Any,
        assignment_matrix: Any,
        subgraph_ids: np.ndarray,
    ) -> None:
        if not sp.issparse(assignment_matrix) and not isinstance(assignment_matrix, np.ndarray):
            raise TypeError('assignment_matrix must be a SciPy sparse matrix or NumPy array.')
        if len(assignment_matrix.shape) != 2:
            raise ValueError('assignment_matrix must be two-dimensional.')
        original_node_count, assignment_columns = assignment_matrix.shape
        if original_node_count <= 0:
            raise ValueError('assignment_matrix must have at least one row.')
        if assignment_columns <= int(subgraph_ids.max()):
            raise ValueError("graph.graph['subgraph_id'] exceeds assignment_matrix columns.")

        if sp.issparse(original_graph):
            if original_graph.shape != (original_node_count, original_node_count):
                raise ValueError(
                    'original_graph adjacency shape must match assignment_matrix rows.'
                )
            return
        if not isinstance(original_graph, nx.Graph):
            raise TypeError('original_graph must be a NetworkX graph or SciPy sparse matrix.')
        if original_graph.number_of_nodes() != original_node_count:
            raise ValueError(
                'original_graph node count must match assignment_matrix rows.'
            )
        if set(original_graph.nodes()) != set(range(original_node_count)):
            raise ValueError(
                'original_graph node ids must be exactly 0..N-1 so they align with '
                'assignment_matrix rows.'
            )

    def split(self, fold: int) -> Dict[str, np.ndarray]:
        """Return local dataset positions for one train/validation/test split."""
        fold_index = _integer(fold, 'fold', 0)
        if fold_index >= len(self._fold_indices):
            raise IndexError(
                'fold must be in [0, {}), got {}.'.format(
                    len(self._fold_indices), fold_index
                )
            )

        test_positions = self._fold_indices[fold_index]
        validation_positions = self._fold_indices[
            (fold_index + 1) % len(self._fold_indices)
        ]
        train_mask = np.ones(len(self.dataset), dtype=bool)
        train_mask[test_positions] = False
        train_mask[validation_positions] = False
        train_positions = np.flatnonzero(train_mask).astype(np.int64, copy=False)

        return {
            'train': train_positions.copy(),
            'val': validation_positions.copy(),
            'test': test_positions.copy(),
        }

    def loaders(self, fold: int, batch_size: int) -> Dict[str, DataLoader]:
        """Build loaders over the already-preprocessed sparse graph tensors."""
        size = _integer(batch_size, 'batch_size', 1)
        positions = self.split(fold)
        generator = torch.Generator()
        generator.manual_seed(self._seed + int(fold))

        return {
            'train': DataLoader(
                Subset(self.dataset, positions['train'].tolist()),
                batch_size=size,
                shuffle=True,
                generator=generator,
                collate_fn=collate_graph_batch,
            ),
            'val': DataLoader(
                Subset(self.dataset, positions['val'].tolist()),
                batch_size=size,
                shuffle=False,
                collate_fn=collate_graph_batch,
            ),
            'test': DataLoader(
                Subset(self.dataset, positions['test'].tolist()),
                batch_size=size,
                shuffle=False,
                collate_fn=collate_graph_batch,
            ),
        }

    def all_loader(self, batch_size: int) -> DataLoader:
        """Return all active subgraphs in their source-index order."""
        size = _integer(batch_size, 'batch_size', 1)
        return DataLoader(
            self.dataset,
            batch_size=size,
            shuffle=False,
            collate_fn=collate_graph_batch,
        )
