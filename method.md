# Method

本文档根据当前代码整理项目的算法原理、整体逻辑与运行步骤。当前仓库主要包含两条相关但用途不同的流程：

1. **主训练流程**：`train.py` 调用 `MISGL.bin.train_eval`，训练 `MISGLEncoder` 完成子图级二分类。
2. **二阶段粗图关系实验**：`coarse_relation_experiment.py` 先从主模型导出每个子图的固定表示 `z_mil`，再把每个子图视为粗图节点，在子图关系图上训练轻量关系模型。

`data_information.md` 的实验记录显示：`ogbn_arxiv` 和 `reddit` 的子图之间确实存在可用关系，但原始 coarse graph 不是稳定最优的关系载体，尤其在 Reddit 上，基于 `z_mil` 的 kNN 图比原始 coarse graph 更干净。因此当前代码既保留了主模型中的可选 coarse position head，也提供了二阶段关系图实验来比较 coarse graph、`z_mil` kNN graph 与不同传播模型。

## 1. 数据抽象

预处理后的数据由 `GraphDataLoaderWrapper` 从 `{data_name}_processed.pkl` 读取。代码不直接读取原始大数据文件，而是假设 pickle 中已经包含：

- `subgraph_structures`：每个样本对应一个 NetworkX 子图。
- `train_test_split`：原始 train/test 索引。
- `subgraph_labels` 或子图自身 `graph['label']`：子图级二分类标签。
- `original_graph`：原始大图。
- `assignment_matrix`：原始节点到子图的分配矩阵 `S`，用于构建子图级 coarse graph。
- `dataset_metadata`：特征维度、最大节点数等元信息。

每个子图 `G_i` 被转换为固定尺寸张量：

```text
X_i: [N_max, F]       padded node features
A_i: [N_max, N_max]  padded adjacency matrix
y_i: scalar          graph/subgraph label
n_i: scalar          valid node count
```

如果开启 Branch-B 结构特征，还会为每个有效节点计算 7 维结构特征：

```text
degree_norm
log_degree_norm
avg_neighbor_degree_norm
2_hop_walk_log_norm
triangle_count_log_norm
clustering_coeff
core_number_norm
```

如果开启 `position_head`，每个子图还会附带粗图字段：

```text
coarse_node_id
coarse_node_num
coarse_neighbor_index: [top_k]
coarse_neighbor_weight: [top_k]
```

## 2. 粗图构建

粗图构建逻辑在 `MISGL/utils/coarse_graph.py` 中。核心思想是把原始图的节点级邻接关系投影到子图级：

```text
C_raw = S^T A S
```

其中：

- `A` 是原始图邻接矩阵。
- `S` 是原始节点到子图的 assignment matrix。
- `C_raw[p, q]` 表示子图 `p` 与子图 `q` 之间由原始图边诱导出的连接强度。

当 `normalize=true` 时，代码按两个端点子图大小做归一化：

```text
C = D_s^-1 C_raw D_s^-1
```

随后执行：

1. 可选移除自环。
2. 可选对称化：`C = max(C, C^T)`。
3. 每行保留权重最大的 `top_k` 条边。
4. 保存或读取 `.cache/position_head` 下的粗图缓存。

这个 coarse graph 有两种用法：

- 主模型中的 `position_head` 用它为当前子图聚合邻居表示。
- 二阶段实验中可直接作为关系图，或被 `z_mil` kNN graph 替代。

## 3. 主模型 MISGLEncoder

主模型定义在 `MISGL/models/encoder.py`，整体结构是：

```text
subgraph nodes
  -> ResidualGATLayer
  -> masked mean pooling h1
  -> optional Branch-B MIL attention => z_mil
  -> optional coarse position head => z_pos
  -> classifier MLP
  -> binary logit
```

### 3.1 子图内 Residual GAT

`ResidualGATLayer` 对每个子图内部做一层多头 GAT。对第 `r` 个 head：

```text
Wh_i = X_i W_r
e_ij = LeakyReLU(a_src^T Wh_i + a_dst^T Wh_j)
alpha_ij = softmax_j(e_ij), j in N(i) union {i}
z_i = sum_j alpha_ij Wh_j
```

实现细节：

- 邻接矩阵会自动加单位阵，因此每个节点包含 self-loop。
- 注意力只在 `adj + I` 指定的边上归一化。
- 多头输出默认 concat。
- 输出加 residual connection；当输入输出维度不一致时使用线性投影。
- padding 节点通过 mask 置零。

GAT 输出为：

```text
H_node: [B, N_max, hidden_dim]
```

### 3.2 Mean Pooling 基线表示

主模型总会先计算一个子图均值表示：

```text
h1_i = (1 / n_i) sum_{v in G_i} H_node[v]
```

当 `branch_b.use=false` 且 `position_head.use=false` 时，分类器直接使用 `h1_i`。

### 3.3 Branch-B: Gated-Attention MIL Head

当 `branch_b.use=true` 时，模型使用 `MILBranchB` 替代简单均值池化，得到 `z_mil`。

对每个有效节点表示 `h_v`，Branch-B 计算 gated attention score：

```text
s_v = w^T (tanh(V phi_v) * sigmoid(U phi_v))
a_v = softmax_{v in G_i}(s_v)
z_h = sum_v a_v h_v
```

其中 `phi_v` 默认是节点表示 `h_v`。如果开启结构特征，则：

```text
g_v = structural_encoder(structural_features_v)
phi_v = concat(h_v, g_v)
```

结构特征聚合为：

```text
z_g = sum_v a_v g_v
```

当前 `b_on.yml` 中使用的结构融合方式是 `gated_residual`：

```text
z_g_proj = Linear(z_g)
gate = sigmoid(MLP(concat(z_h, z_g_proj)))
z_B = z_h + residual_scale * gate * z_g_proj
```

最终：

```text
z_mil = z_B
```

Branch-B 的作用是让模型在一个子图内部学习“哪些节点更重要”，并在可选结构特征的辅助下形成更稳的子图级表示。

### 3.4 Position Head: 粗图上的残差 GCN 表示

当 `position_head.use=true` 时，模型额外使用 `ResidualGCNPositionHead` 在子图级 coarse graph 上聚合邻居信息。

Position head 维护一个 `coarse_memory`：

```text
M[p] = 当前已知的 coarse node p 的 h1 表示
```

前向时，当前 batch 的 `h1` 会先写入 memory，然后根据 `coarse_neighbor_index` 和 `coarse_neighbor_weight` 读取邻居表示：

```text
m_i = sum_j normalized_weight_ij * M[j]
message_i = Dropout(ReLU(W m_i))
z_pos_i = h1_i + residual_scale * message_i
```

如果当前 batch 中某个邻居正好也在 batch 内，代码会用当前 batch 的最新 `h1` 替代 memory 中的旧值。

开启 position head 后，分类器输入为：

```text
H_i = concat(z_mil_i, z_pos_i)
```

这里的 `z_mil_i` 是 Branch-B 输出；如果未开启 Branch-B，则退化为 `h1_i`。

### 3.5 二分类器

分类器是两层 MLP：

```text
logit_i = Linear2(Dropout(LeakyReLU(Linear1(classifier_input_i))))
```

输出是一个二分类 logit，评估时通过：

```text
p_i = sigmoid(logit_i)
pred_i = 1 if p_i > 0.5 else 0
```

## 4. 损失函数

损失函数在 `MISGL/utils/get_loss.py`。

基础损失是 binary cross entropy with logits：

```text
L_bce = BCEWithLogits(logit, y_smooth)
```

可选 label smoothing：

```text
y_smooth = y * (1 - smoothing) + 0.5 * smoothing
```

如果开启 Branch-B 且 `branch_b.attention_shape_loss_weight > 0`，会加一个注意力形状正则项。代码计算每个子图内注意力分布的归一化熵：

```text
entropy_i = - sum_v a_v log(a_v) / log(num_nodes_i)
```

正样本希望注意力更集中，负样本希望注意力更分散：

```text
L_shape_i = entropy_i          if y_i = 1
L_shape_i = 1 - entropy_i      if y_i = 0
L = L_bce + lambda * mean(L_shape)
```

当前 `config/b_on.yml` 中 `attention_shape_loss_weight=0.05`。

## 5. 主训练与评估流程

### 5.1 入口与配置

`train.py` 做以下工作：

1. 读取 YAML 超参数。
2. 合并 `hparams_lib.DEFAULT_HPARAMS` 默认值。
3. 可选自动选择空闲 GPU。
4. 设置随机种子。
5. 遍历 `data_name_set` 中的数据集。
6. 为每个数据集调用 `train_eval.train_eval`。

注意：`MISGL/bin/train_eval.py` 文件末尾将：

```text
train_eval = fixed_cv_train_eval
```

因此当前 `train.py` 默认实际运行的是固定 10 折 CV，而不是 repeated holdout。

### 5.2 固定 10 折 CV

固定 CV 分割由 `prepare_cv_split.py` 或训练时已有 manifest 提供。协议是：

```text
10 folds
test_fold = fold_idx
val_fold = (test_fold + 1) mod 10
train_folds = remaining 8 folds
```

分割构建时优先使用 group-aware stratified split：

- 如果数据中存在 `group_ids`、`subject_ids`、`patient_ids` 等字段，则按组划分，避免同组样本泄漏。
- 如果没有组信息，则每个图退化为独立组。
- 标签分布尽量分层。

每折训练一个新的 `MISGLEncoder`。

### 5.3 单折训练循环

`train_eval_iter` 执行按 epoch 的训练：

1. 设置 Adam 优化器。
2. 遍历 train loader。
3. 前向得到 logit 或 `{'ypred_A': logit, ...}`。
4. 计算 `fused_loss`。
5. 反向传播。
6. 按 `grad_clip` 裁剪梯度。
7. optimizer step。
8. 在 val loader 上评估 loss 与 accuracy。
9. 根据 val accuracy 选择最优模型；若 accuracy 相同，则用 val loss 打破平局。
10. 若连续 `patience` 个 epoch 无提升，则 early stop。
11. 训练结束后恢复 val 最优权重。

### 5.4 Position Memory 的评估处理

如果模型开启 `position_head`，评估时不能直接沿用训练过程残留的 memory。`evaluate` 会：

1. snapshot 当前 position memory。
2. reset memory。
3. 先遍历当前评估 split，用 `update_position_memory_from_batch` 填充该 split 的 coarse memory。
4. 再正式前向评估。
5. 评估结束后 restore 原始 memory。

这样可以避免不同 split 之间的 memory 状态污染。

### 5.5 评估指标与结果保存

主训练评估指标包括：

```text
accuracy
precision
recall
F1
optional loss
```

固定 10 折结束后，代码会统计每个 split 的均值与标准差，并保存：

- `{timestamp}_cv_results.json`
- 可选 Excel 汇总表与 leaderboard
- 可选 Branch-B attention 分析 Excel

## 6. 二阶段粗图关系实验

`coarse_relation_experiment.py` 用于回答一个独立问题：

> 当每个子图已经有一个固定表示 `z_mil` 后，子图之间的关系图是否还能提升分类？

它不是 `train.py` 默认主流程的一部分，而是额外实验入口。

更具体地说，这个文件做的是一个**二阶段子图关系建模对照实验**：

```text
Stage 1: 训练/加载 MISGL 子图编码器
  -> 为每个子图导出固定表示 z_mil

Stage 2: 把每个子图当成一个 coarse node
  -> 构建子图级关系图
  -> 在固定 z_mil 上训练轻量关系模型
  -> 比较“只看自身表示”和“引入子图关系”是否更好
```

它主要验证四件事：

1. **固定子图表示本身是否已经足够强**  
   `mlp` 模型只使用 `z_mil`，不使用任何子图间边，因此它是二阶段实验的核心基线。

2. **原始 coarse graph 是否适合传播**  
   当 `relation_graph=coarse` 时，脚本使用由 `S^T A S` 构建的原始子图关系图，测试 GCN、APPNP、SAGE 等传播模型是否能利用这批边提升分类。

3. **基于表示空间的 kNN 图是否比原始 coarse graph 更可靠**  
   当 `relation_graph=zmil_knn` 时，脚本用固定 `z_mil` 的 cosine 相似度构建 kNN 图，再训练同一批关系模型。这个设置用于检验：真正有用的子图关系是否更体现在 learned representation space 中，而不是原始粗图边中。

4. **不同传播机制对噪声关系图的敏感性**  
   `gcn` 和 `appnp` 更偏强传播/平滑；`sage` 显式保留自身表示；`gated_sage` 进一步学习 node-level gate。通过同一 split、同一 `z_mil`、同一关系图下的对比，可以判断性能变化来自关系建模方式，而不是 Stage 1 表示差异。

因此，`coarse_relation_experiment.py` 的实验对象不是“重新训练一个完整端到端模型”，而是把 Stage 1 的子图编码结果冻结下来，专门隔离研究**子图间关系图和传播模型**的贡献。

### 6.1 Stage 1: 训练或加载子图表示模型

Stage 1 可以：

1. 重新训练 `MISGLEncoder`。
2. 加载已有 checkpoint。
3. 直接复用已有 embeddings 文件。

正式实验一般使用 Branch-B 输出的：

```text
z_mil
```

作为每个子图的固定节点特征。配置中建议 `stage1_use_position_head=false`，因为二阶段本身就是为了单独分析子图间关系。

导出的 payload 包含：

```text
features: [num_subgraphs, dim]
labels: [num_subgraphs]
orig_indices
coarse_node_ids
stage1_logits
embedding_key
```

### 6.2 Stage 2: 构建关系图

二阶段把每个子图看成一个 coarse node。关系图有两种来源：

1. `coarse`：直接使用 `S^T A S` 构建的原始粗图。
2. `zmil_knn`：在固定 `z_mil` 表示空间中构建 kNN 图。

`zmil_knn` 的构建方式：

```text
x_i = normalize(z_mil_i)
sim_ij = cosine(x_i, x_j)
N_k(i) = top-k most similar nodes excluding i
```

边权可选：

```text
binary:          w_ij = 1
cosine_shift:    w_ij = 0.5 * (sim_ij + 1)
positive_cosine: w_ij = max(sim_ij, 0)
```

当前配置中使用：

```text
relation_graph = zmil_knn
knn_k = 16
knn_weight_mode = positive_cosine
knn_symmetrize = true
```

这与 `data_information.md` 中的诊断一致：两个数据集上 `z_mil` kNN graph 的 same-label 比例通常高于原始 coarse graph。

### 6.3 特征标准化与邻接归一化

二阶段训练前会对 `z_mil` 特征做标准化：

```text
standard: 使用 train split 的 mean/std 标准化所有节点
l2:       对每个节点表示做 L2 normalize
none:     不归一化
```

邻接矩阵根据模型类型使用两种归一化：

- GCN/APPNP 使用带 self-loop 的 symmetric normalization：

```text
A_norm = D^-1/2 (A + I) D^-1/2
```

- SAGE/GatedSAGE 使用 row normalization，且不额外加 self-loop：

```text
A_row = D^-1 A
```

### 6.4 二阶段关系模型

二阶段支持以下模型。

#### MLP

只使用自身 `z_mil`，不使用关系图：

```text
logit_i = MLP(z_i)
```

它是“固定子图表示本身有多强”的基线。

#### GCN

两层 GCN 风格传播：

```text
h = ReLU(W1 A_norm Z)
logit = W2 A_norm h
```

它会强制混合邻居，因此当关系图噪声较大时容易过平滑或引入错误邻居信号。

#### APPNP

先用 MLP 得到初始 logit，再做 Personalized PageRank 风格传播：

```text
logit_0 = MLP(z_i)
logit^{t+1} = (1 - alpha) A_norm logit^t + alpha logit_0
```

它保留一部分自身预测，但仍会把邻居预测传播进来。

#### SAGE

GraphSAGE 风格显式保留自身表示，并拼接邻居均值：

```text
neigh_i = A_row Z
h_i = ReLU(W1 concat(z_i, neigh_i))
neigh_h_i = A_row H
logit_i = W2 concat(h_i, neigh_h_i)
```

相比 GCN/APPNP，SAGE 对噪声关系图更稳，因为它没有把自身表示完全淹没在邻居平滑里。

#### GatedSAGE

GatedSAGE 在 SAGE 基础上加 node-level gate，学习“该节点整体上听多少邻居”：

```text
neigh_i = A_row Z
gate1_i = sigmoid(MLP(concat(z_i, neigh_i, |z_i - neigh_i|)))
h_i = ReLU(W1 concat(z_i, gate1_i * neigh_i, gate1_i))

neigh_h_i = A_row H
gate2_i = sigmoid(MLP(concat(h_i, neigh_h_i, |h_i - neigh_h_i|)))
logit_i = W2 concat(h_i, gate2_i * neigh_h_i, gate2_i)
```

该模型记录 `gate1` 和 `gate2`，供诊断脚本分析。但它是 node-level gate，不是 edge-level gate；也就是说它先平均邻居，再决定整体听多少邻居，不能逐边筛掉坏邻居。

### 6.5 二阶段训练

每个关系模型的训练步骤：

1. 按 train/val/test mask 划分节点。
2. 用 AdamW 优化。
3. 只在 train mask 上计算 BCEWithLogits loss。
4. 每个 epoch 在 train/val/test 上全图评估。
5. 根据 `selection_metric` 选择最优模型，默认是 `val_roc_auc`。
6. 连续 `relation_patience` 个 epoch 无提升则 early stop。
7. 保存每个模型的 checkpoint、`relation_results.json`、`relation_predictions.csv`。

二阶段指标包括：

```text
loss
accuracy
precision
recall
F1
ROC-AUC
Average Precision
positive_rate
pred_positive_rate
```

## 7. 诊断逻辑

`coarse_relation_diagnostics.py` 不重新训练模型，而是读取二阶段输出和 embeddings，对关系图质量进行诊断。

主要诊断包括：

1. **边可靠性**：统计 coarse edge 两端是否同标签。
2. **边特征 AUC**：比较 `coarse_weight`、`z_mil_cosine`、预测概率一致性对 same-label 的解释能力。
3. **分桶统计**：观察高权重/高相似边是否更同标签。
4. **节点邻居统计**：计算每个节点的邻居同标签率、加权邻居正类比例、邻居预测均值等。
5. **改对/改错分析**：以 `mlp` 为基线，统计关系模型将哪些样本改对或改错。
6. **gate 诊断**：统计 `gated_sage` 的 `gate1/gate2` 是否能区分可靠邻居。
7. **kNN vs coarse graph**：比较 `z_mil` kNN 图和原始 coarse graph 的 same-label 比例与边重合度。

`data_information.md` 中记录的当前结论是：

- `ogbn_arxiv` 的原始 coarse graph 有一定同质性，但 `z_mil` 相似度和预测概率一致性更能判断边是否可靠。
- `reddit` 的原始 coarse graph 噪声更大，原始边权几乎不能解释边是否同标签。
- `z_mil` kNN graph 在两个数据集上都比原始 coarse graph 更干净，尤其 Reddit 上差距明显。
- `mlp(z_mil)` 已经很强，强传播模型如 GCN/APPNP 可能把低质量邻居信号混入自身强表示。
- SAGE 更稳定，但提升较小。
- 当前 `gated_sage` 是 node-level gate，诊断显示它不足以可靠地区分好邻居和坏邻居。

## 8. 当前代码中的方法定位

从算法贡献角度，当前代码可以理解为三层：

1. **子图内表示学习**  
   使用 residual multi-head GAT 编码每个子图内部节点结构。

2. **子图级 MIL 聚合**  
   使用 Branch-B gated attention 从节点表示中学习子图表示 `z_mil`，并可融合局部结构统计。

3. **子图间关系建模**  
   有两种实现路径：
   - 主模型中的 `position_head`：在线使用 coarse graph 聚合邻居 `h1`，并与 `z_mil` 拼接分类。
   - 二阶段关系实验：离线固定 `z_mil`，比较 MLP、GCN、APPNP、SAGE、GatedSAGE，以及 coarse graph 与 `z_mil` kNN graph。

当前实验记录支持的主要方向是：与其继续单纯调强传播模型，不如优先改进关系图本身，例如使用 `z_mil` kNN graph、基于 `z_mil` 相似度过滤 coarse edge，或进一步发展 edge-level gate。

## 9. 推荐运行顺序

### 9.1 生成固定 CV 划分

```bash
python prepare_cv_split.py --hparam_path ./config/b_on.yml
```

### 9.2 训练主模型

开启 Branch-B 和 position head：

```bash
python train.py --hparam_path ./config/b_on.yml
```

只训练基础 GAT + mean pooling/MLP：

```bash
python train.py --hparam_path ./config/b_off.yml
```

### 9.3 运行二阶段关系实验

使用配置中的 coarse relation 实验参数：

```bash
python coarse_relation_experiment.py --hparam_path ./config/b_on.yml
```

如果只想跑某一折：

```bash
python coarse_relation_experiment.py --hparam_path ./config/b_on.yml --fold_idx 0
```

### 9.4 运行诊断

```bash
python coarse_relation_diagnostics.py --hparam_path ./config/b_on.yml
```

## 10. 关键文件索引

- `train.py`：主训练入口、配置读取、多数据集循环、GPU 自动选择。
- `prepare_cv_split.py`：固定 10 折 CV manifest 生成。
- `MISGL/bin/train_eval.py`：训练、早停、固定 CV、评估汇总、注意力导出入口。
- `MISGL/models/encoder.py`：主模型 `MISGLEncoder`。
- `MISGL/layers/gat_layer.py`：Residual multi-head GAT。
- `MISGL/models/mil_head.py`：Branch-B gated-attention MIL 聚合。
- `MISGL/models/position_head.py`：coarse graph residual GCN position head。
- `MISGL/utils/load_data.py`：数据加载、子图张量化、结构特征、CV split、coarse graph 附加。
- `MISGL/utils/coarse_graph.py`：`S^T A S` 粗图构建与缓存。
- `MISGL/utils/get_loss.py`：BCE、label smoothing、MIL attention shape loss。
- `MISGL/utils/evaluate.py`：二分类评估和 position memory 评估处理。
- `coarse_relation_experiment.py`：二阶段 `z_mil` 固定表示与关系模型实验。
- `coarse_relation_diagnostics.py`：粗图可靠性、kNN、gate、改对/改错诊断。
- `attention_analyzer.py`：Branch-B 注意力导出与分析。
- `data_information.md`：当前数据特性与粗图关系诊断记录。
