# Multiple-Instance Subgraph Learning

Code for the paper *Multiple-Instance Subgraph Learning*, implementing
Node-Attentive Subgraph Learning (NASGL).

## Installation and training

Use Python 3.10.18. Run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m data_construction.synthetic_data --data-root data
python train.py --benchmark syn3 --data-root data --device cpu
python evaluate.py --run-dir outputs/paper/syn3/nasgl --device cpu
```

Use `--device cuda --cuda-device 0` for GPU training. Training settings are in
`MISGL/config.py`. Real partitioned data belong in
`data/<dataset>/{bfs,metis,metis_perturb}/`.

## Partitioned datasets

The experiment data are available in [Releases](https://github.com/jasmineyg/MISGL/releases/tag/paper-data-v1).
With GitHub CLI installed, run from the repository root to download, verify and
restore the original `data/` directory:

```bash
gh release download paper-data-v1 -R jasmineyg/MISGL -p 'misgl-data.tar.gz.part-*' -p '*SHA256SUMS' -D /tmp/misgl-data
(cd /tmp/misgl-data && shasum -a 256 -c SHA256SUMS)
cat /tmp/misgl-data/misgl-data.tar.gz.part-* | tar -xzf -
shasum -a 256 -c /tmp/misgl-data/DATA_SHA256SUMS
```

## Real-data construction

Use a separate Python 3.9.23 environment with native METIS 5.2.1. The same
requirements file selects the dependencies for each Python version.

```bash
conda create -n misgl-data -c conda-forge python=3.9.23 metis=5.2.1
conda activate misgl-data
python -m pip install -r requirements.txt
python -m data_construction.real_data --benchmark igb_tiny__metis --raw-dir /path/to/IGB-tiny --data-root data
```

Dataset settings are in `data_construction/dataset_settings.py`.
Data and generated outputs are excluded from version control.
