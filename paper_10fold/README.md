# MISGL paper: fixed single 10-fold protocol

This workflow replaces the abandoned repeated 5x10 run.

- One grouped/stratified 10-fold split, seed `1024`, adjacent validation fold.
- Exactly two Stage-1 fits per dataset/fold: Mean and MIL.
- Stage-2 always consumes detached cached Stage-1 embeddings.
- Stage-1 checkpoints, all-bag embeddings, logits, probabilities, predictions,
  labels and original indices are cached.
- MIL attention is exported only for positive test bags. Attention analysis is
  out-of-fold and never starts model training.
- Stage-2 protocol: 300 epochs, patience 50, learning rate 1e-3, weight decay
  5e-4, one-layer coarse GCN, top-k 16 coarse adjacency.

`build_reuse_manifest.py` records the exact source of every reused artifact and
the missing stages that are allowed to run.

