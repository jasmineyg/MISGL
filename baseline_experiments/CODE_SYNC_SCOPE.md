# Local/remote experiment-code synchronization scope

The final synchronization audit covers every source, launcher, test, and
documentation file used by the formal comparison runs. Data, caches, checkpoints,
logs, results, backups, IDE metadata, `.git`, and pre-existing unused upstream
files are excluded from the code hash comparison.

## Attention-based MIL

- Local: `D:/AResearch/Code/AttentionDeepMIL`
- Remote: `/data/yg/Subgraph-MIL/AttnMIL`
- Files: `README.md`, `data_graph.py`, `main.py`, `model.py`,
  `fair_protocol.py`, `fair_experiment.py`, `launch_fair_experiments.sh`

## RGMIL

- Local: `D:/AResearch/RGMIL`
- Remote: `/data/yg/Subgraph-MIL/RGMIL`
- Files: `README.md`, `fair_protocol.py`, `rgmil_fair_model.py`,
  `fair_experiment.py`, `launch_fair_experiments.sh`

## SubGNN

- Local: `D:/AResearch/Code/SubGNN`
- Remote: `/data/yg/Subgraph-MIL/SubGNN2`
- Files: `README.md`, `config.py`, `launch_fair_experiments.sh`,
  `SubGNN/SubGNN.py`, `SubGNN/anchor_patch_samplers.py`,
  `SubGNN/data_loading.py`, `SubGNN/fair_experiment.py`,
  `SubGNN/fair_protocol.py`, `SubGNN/gamma.py`,
  `SubGNN/subgraph_mpn.py`, `SubGNN/subgraph_utils.py`,
  `tests/test_dtw_padding.py`, `tests/test_noncontiguous_anchor_mask.py`, and
  `tests/test_batch_shortest_paths.py`.

The final `code_sync_manifest.json` records SHA-256 hashes from both sides and
requires every listed pair to match.
