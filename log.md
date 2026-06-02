# 实验日志

## 记录格式

每次追加日志时，统一按下面的要点写，避免记录过长或格式不一致：

- **分支名**：例如 `exp/structure`
- **时间**：例如 `2026-05-29`
- **主要改动**：一句话说明核心改法，不展开过长实现细节
- **改进结果**：记录主要数据集上的效果变化
- **存在问题**：记录退化、风险或下一步需要验证的点

---

## `exp/structure`，2026-05-26 15:42

- **主要改动**：在 Branch B 的 gated-attention MIL 聚合中加入结构统计量。
- **改进结果**：未单独记录。
- **存在问题**：当时结构统计量只包含 degree 相关特征，结构信息只参与 attention 打分，bag 表示仍只由原始 GAT 节点表示 `h_i` 加权得到。
- **说明**：若 `branch_b.use_structural_features=true`，模型从邻接矩阵计算 `degree_norm`、`log_degree_norm`、`avg_neighbor_degree_norm`、`2_hop_walk_log_norm`，并与节点表示拼接后送入 Branch B 的 gated attention scorer。

## `exp/structure`，2026-05-28

- **主要改动**：继续扩展 Branch B 的结构统计量，追加局部形态特征。
- **改进结果**：未单独记录。
- **存在问题**：结构特征仍只参与 attention scorer 的输入 `[h_i, g_i]`，没有直接进入 bag 表示。
- **说明**：新增 `triangle_count_log_norm`、`clustering_coeff`、`core_number_norm`。其中 triangle count 和 clustering coefficient 基于子图内无向二值邻接矩阵计算闭合三元结构，core number 通过 bag 内部 k-core peeling 得到并按 `n - 1` 归一化。

## `main/stru-attn`，2026-05-29

- **主要改动**：实现 Branch B concat 版本，将结构统计量编码后直接并入 bag 表示。
- **改进结果**：`ogbn_arxiv` 准确率倒退，`reddit` 有改进。
- **存在问题**：`z_B = concat(z_h, z_g)` 对数据集不够稳定，分类器被迫直接接收结构摘要；后续改为 gated residual，以保留退回 `z_h` 的路径。
- **说明**：7 维结构统计量 `g_i` 经过 `MLP_struct(7 -> 32 -> 32)` 得到结构 embedding `e_i`；attention scorer 输入由 `[h_i, g_i]` 改为 `[h_i, e_i]`；bag 表示同时聚合 `z_h = sum_i a_i h_i` 和 `z_g = sum_i a_i e_i`。

## `main`，2026-06-02

- **主要改动**：将当前结构注意力实验线迁为真正的 `main`，并保留 Branch B 的 gated residual 结构融合与 MIL attention 形状约束。
- **改进结果**：本次只整理主线和记录逻辑，未记录完整训练结果；当前 loss 分支已通过 one-batch forward/backward smoke test。
- **存在问题**：`attention_shape_loss_weight=0.05` 仍需完整交叉验证确认；正包尖锐、负包平均的熵约束可能影响分类 BCE，需要观察不同数据集稳定性。
- **说明**：当前 Branch B 使用 7 维结构统计量，经结构 MLP 得到 `e_i` 后参与 attention scorer；bag 表示以 `z_h = sum_i a_i h_i` 为主，并通过 `z_B = z_h + structural_residual_scale * gate([z_h, proj(z_g)]) * proj(z_g)` 注入结构摘要。loss 由 BCE 扩展为 `BCE + attention_shape_loss_weight * L_attn`，其中正包最小化归一化 attention entropy，负包最小化 `1 - entropy`，单节点 bag 跳过。
