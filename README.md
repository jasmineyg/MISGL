# MISGL

MISGL trains a sparse graph encoder with two explicit optional heads:

- **MIL-HEAD** combines seven precomputed structural descriptors with node
  representations, then aggregates instances using gated attention.
- **POS-HEAD** refines the MIL representation over the sparse top-`k`
  subgraph-relation graph. It depends on MIL-HEAD.

Training uses one strict configuration file, `config/train.yml`. Every field is
required. Unknown fields, missing fields, invalid values, and illegal head
combinations stop immediately; the loader does not search for another
configuration or infer missing values.

## Head modes

Only these three modes are legal:

| Mode | MIL-HEAD | POS-HEAD | Command |
| --- | --- | --- | --- |
| Mean-pool baseline | off | off | `python train.py --no-mil-head --no-pos-head` |
| MIL | on | off | `python train.py --mil-head --no-pos-head` |
| MIL + POS | on | on | `python train.py --mil-head --pos-head` |

`POS-HEAD=on, MIL-HEAD=off` is invalid and raises an error before training.
The checked-in YAML enables both heads, so the default run is:

```bash
python train.py --config config/train.yml
```

## Common overrides

```bash
# Train more than one dataset.
python train.py --datasets ogbn_arxiv reddit

# Use CPU and write results to another directory.
python train.py --device cpu --output-dir results/cpu

# Use another strict YAML file.
python train.py --config config/experiment.yml
```

CLI values replace only the corresponding YAML values. Dataset names are
space-separated. Edit `cuda_device` in the YAML when a specific visible CUDA
device is required.
Set `data_dir` to the local directory containing `<dataset>_processed.pkl`
before the first run.

## Computation

Subgraphs are batched as one disconnected sparse graph. GAT attention is
computed only on existing edges and self-loops, using `O(H(E + N))` memory and
work per batch instead of materializing `B x H x N x N` attention tensors.
The seven structural descriptors are calculated once while loading each
dataset and are reused across all epochs and folds.

## Configuration layout

```text
datasets, run_name, data_dir, output_dir, device, cuda_device, seed, folds
|-- model       graph encoder and classifier dimensions
|-- training    optimizer, loss, clipping, and early stopping
|-- mil_head    MIL switch, structure fusion, and attention settings
`-- pos_head    POS switch and relation-propagation settings
```

The public configuration API is in `MISGL/config.py`. The root entry point
only parses CLI overrides and calls `MISGL.trainer.run(config)`.
