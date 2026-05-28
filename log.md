# 实验日志

exp/structure（2026-05-26 15:42）：
在 Branch B 的 gated-attention MIL 聚合里加入结构统计量。
当前流程是：每个子图 GAT 得到节点表示 `h`，再按有效节点 mask 展平。若 `branch_b.use_structural_features=true`，模型会从邻接矩阵计算每个节点的 `degree_norm`、`log_degree_norm`、`avg_neighbor_degree_norm`、`2_hop_walk_log_norm`，并与节点表示拼接后送入 Branch B 的 gated attention 打分。attention 权重用于对原始节点 embedding 加权求和得到 bag 表示 `z_B`。

exp/structure（2026-05-28）：
继续扩展 Branch B gated-attention 的结构统计量，在已有 degree 相关特征后追加局部形态特征：`triangle_count_log_norm`、`clustering_coeff`、`core_number_norm`。其中 triangle count 和 clustering coefficient 基于子图内的无向二值邻接矩阵计算闭合三元结构，core number 通过每个 bag 内部的 k-core peeling 得到并按 `n - 1` 归一化。结构特征只参与 attention scorer 的输入 `[h_i, g_i]`，bag 表示仍由 attention 权重对原始 GAT 节点表示 `h_i` 加权得到。
