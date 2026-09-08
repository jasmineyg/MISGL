# MISGL

The core model uses all nine structural statistics and a single structural
encoder: `Linear(9,32) -> LayerNorm(32) -> ReLU`.

## Structural representation

Features are computed once per dataset on unweighted, undirected subgraphs
with self-loops removed, in this fixed order:

1. Normalized degree
2. Normalized average neighbor degree
3. Exactly-two-hop reachability ratio
4. Personalized PageRank self-return probability (restart probability 0.15)
5. Log-normalized triangle count
6. Clustering coefficient
7. Normalized core number
8. PageRank normalized by its maximum
9. Absolute Fiedler-vector coordinate normalized by its maximum (LapPE)

All nine inputs are used. There is no feature mask, alternative structural
profile, two-layer encoder, or structural dropout option. The structural output
width is fixed at 32. The node GAT output is 256 by default, giving a 288-wide
attention input. Attention uses parallel tanh/sigmoid projections with hidden
width 128. The same attention weights pool semantic and structural embeddings;
a 32-to-256 projection and a 64-hidden-unit gate fuse structure into the semantic
representation. The final MIL representation remains 256-dimensional.

The GAT, MIL/POS head switches, losses and two-stage training are retained.
POS uses the frozen MIL embeddings and the subgraph relation graph.

## Run

Use Python with PyTorch, NumPy, SciPy, NetworkX, scikit-learn, ruamel.yaml and tqdm.
Every field in `config/train.yml` is required; unknown fields are rejected.
Set `data_dir` to a directory containing `<dataset>_processed.pkl` before running.

```bash
# Default: MIL + POS
python train.py --config config/train.yml
# MIL only
python train.py --mil-head --no-pos-head
# Mean pooling baseline
python train.py --no-mil-head --no-pos-head
# Override datasets and output directory
python train.py --datasets ogbn_arxiv reddit --output-dir results/full9
```

POS requires MIL. Structural statistics are label-free and precomputed while
loading data. GAT processes disconnected sparse graph batches. Descriptor
calculation includes dense subgraph matrix operations and eigendecomposition.
Use a new output directory for new training; previous masked or differently sized
models must not be treated as results from this configuration. Old YAML files
containing removed structural options must be updated to the checked-in schema.

## Verify

```bash
python -m unittest discover -s tests
```

Only the core package, configuration, entry point and tests are maintained here.
Historical experiment scripts and result records remain outside Git tracking.
