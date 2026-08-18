# Baseline comparison protocol

## Shared evaluation protocol

- Data and labels: the exact processed datasets used by MISGL. Each processed
  subgraph is one bag and its stored bag label is the target.
- Splits: immutable grouped-stratified 10-fold manifests from
  `paper_10fold_20260814`; fold `f` is test, fold `(f + 1) % 10` is validation,
  and the remaining eight folds are training.
- Seed: 1024 for split generation and model runs. No baseline regenerates or
  reshuffles the fold membership.
- Model selection: validation loss only. Test folds are evaluated once from the
  checkpoint selected on the corresponding validation fold.
- Prediction rule: positive iff `probability > 0.5` (an exact tie is negative).
- Metrics: ACC, precision, recall/sensitivity, specificity, binary F1, macro F1,
  balanced accuracy, ROC-AUC, PR-AUC, and confusion counts. Final dispersion is
  the sample standard deviation across the ten test folds.
- Reproducibility artifacts: resolved config, checkpoint, epoch history, and
  test labels/probabilities/predictions/original indices are retained per fold.

## Attention-based MIL (Ilse et al., 2018)

The baseline retains gated embedding-level attention and treats graph nodes as
unordered instances; all source graph edges are intentionally ignored. The
formal run uses a 128-dimensional instance embedding, 128-dimensional gated
attention, BCE loss, Adam (`lr=1e-3`, `weight_decay=1e-4`), batch size 128, at
most 200 epochs, and validation-loss early stopping with patience 30.

The paper selected the attention dimension from 64/128/256 using validation
performance. We use the official-code default 128 rather than test-set tuning.

## RGMIL (Zhao et al., 2024)

The input dataset's graph edges are ignored, while RGMIL's own learned bag graph
is retained. Pairwise bag similarity is `exp(-EuclideanDistance)`, following the
paper rather than the identity/cosine placeholder found in one existing copy.
Two VDN agents select the similarity threshold (0.05--1.00) and GAT depth
(1--10) on fold 0 only; the selected action is reused for folds 1--9.

Formal hyperparameters follow the paper/official implementation: BCE loss,
Adam (`lr=5e-4`, `weight_decay=1e-3`), dropout 0.2, LeakyReLU slope 0.1,
discount 0.95, epsilon 1.0 decayed over the first 50 steps, replay/history
lengths 20/10, at most 10,000 search steps, and at most 10,000 supervised
epochs with validation-loss patience 20.

Correctness fixes restore raw logits for BCE, disable dropout at evaluation,
copy similarity matrices before thresholding, and construct transitions for the
actual eight training blocks. They do not alter the method's intended model.

## SubGNN (Alsentzer et al., 2020)

The full SubGNN architecture is retained: neighborhood, position, and structure
channels, each with internal and border subchannels; triangular random-walk
anchors and DTW degree-sequence similarity are used. The supplied node features
replace dataset-specific pretrained GIN embeddings that are unavailable for the
MISGL datasets. This is an input adaptation, not a model-structure change.

The formal run uses one SubGNN layer, eight anchors per channel, a 64/32 FFN,
dropout 0.2, cross-entropy loss, Adam (`lr=1e-3`), batch size 32, at most 200
epochs, and validation-loss early stopping with patience 30.

Scalability fixes preserve exact semantics while avoiding dense all-pairs
shortest-path and dense border-adjacency materialization. On-demand exact
bidirectional source-set BFS, component-grouped queries, sparse border
membership, and public tensor `index_add` replace obsolete/private framework
calls. Regression tests compare optimized border and shortest-path results
against NetworkX on small graphs and real arxiv subgraphs.

The formal radius-one border operation uses the exact adjacency-union identity
instead of constructing one induced `networkx.ego_graph` per component node.
Connected components, radius-one border memberships, and internal/external
degree sequences are cached once per dataset with a source-file fingerprint,
then indexed by the immutable fold manifests. These quantities depend only on
the base graph and bag node membership, never on labels or fold assignment.
Structure anchors and their similarities remain fold- and seed-specific.

Within training, all anchor targets for one component are resolved by one exact
multi-source BFS rather than independent searches. The resulting scalar
distances are cached by split, local subgraph index, component index, and anchor
set for subsequent epochs. The cache key includes the anchor set, so resampled
anchors cannot reuse stale distances. A real Syn1 smoke run after these changes
confirmed finite/decreasing loss, exact output length and test indices, and no
split overlap. Neighborhood-internal anchor distances are analytically zero,
and radius-one neighborhood-border anchor distances are analytically one by
construction. The eight global external-position anchors each use one exact
single-source BFS array shared by all components. Fourteen focused regression
tests, including NetworkX distance comparisons for all analytic/cached modes,
passed on the remote runtime.
On three real arxiv subgraphs, the final bidirectional search matched both
NetworkX and SciPy distances exactly and was 27.7--43.8 times faster than the
one-direction source-set BFS used in the first scalable adapter.
Padded connected-component slots are excluded from DTW and unexpected empty
degree sequences receive zero similarity; valid non-empty sequences retain the
original `fastdtw` value. A dedicated regression test covers both the empty-input
boundary and numerical equivalence on valid inputs.
Sliced anchor masks use `reshape` rather than `view`, preserving logical
element order while supporting non-contiguous PyTorch tensors; a second
regression test executes the actual message-passing edge-index method.

## Pre-flight validation

Each adapter passed a Syn1 smoke run before the formal experiments. The checks
verify disjoint train/validation/test indices, full fold coverage, bag-label
alignment, finite loss, output shape, saved prediction indices, checkpoint
creation, and an observed training-loss decrease. Final aggregation reopens
every `test_predictions.npz`, verifies its original indices against the fixed
test fold, and recomputes all metrics independently.
