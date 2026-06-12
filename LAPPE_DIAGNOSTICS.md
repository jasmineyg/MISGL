# LapPE 诊断总结

本文档汇总当前围绕 Laplacian Positional Encoding（LapPE）的全部诊断现象、
实验结果和建模判断。

## 1. 实验背景

当前模型流程为：

```text
子图内部编码
  -> 得到子图表示 z_mil
  -> 在子图级 coarse graph 上计算 LapPE
  -> PositionMLP(LapPE)
  -> 与 z_mil 拼接后分类
```

实验中观察到：

```text
z_mil baseline ≈ z_mil + LapPE
```

因此诊断重点不是继续修改模型，而是回答：

1. LapPE 本身是否包含标签信息？
2. LapPE 空间是否存在类别聚集？
3. LapPE 邻域是否优于或接近 `z_mil` 邻域？
4. coarse graph 是否健康？
5. top-k 是否破坏谱结构？
6. LapPE 是否提供超出 `z_mil` 的增量信息？
7. 融合器是否实际使用 LapPE？

## 2. 已有诊断数据

当前分析使用了：

- `ogbn_arxiv` 和 `reddit` 的完整基础 LapPE 诊断；
- `ogbn_arxiv` 10 折 `z_mil` 补充结果；
- `reddit` 4 折 `z_mil` 补充结果；
- 未获得原端到端 LapPE 模型 checkpoint。

因此可以判断 LapPE 的信息质量、谱结构、邻域质量和轻量融合增量，但无法对原始
端到端模型做严格的 zero/shuffle LapPE 消融。

## 3. LapPE 是否包含标签信息

LapPE-only Logistic Regression probe：

| dataset | ACC | F1 | AUC | dummy AUC |
|---|---:|---:|---:|---:|
| ogbn_arxiv | 0.8393 | 0.8329 | 0.9202 | 0.4875 |
| reddit | 0.6823 | 0.6269 | 0.7289 | 0.4880 |

置乱 LapPE 后：

- `ogbn_arxiv` AUC 平均下降 `0.4099`；
- `reddit` AUC 平均下降 `0.2368`。

结论：

- LapPE 不是随机噪声。
- arxiv LapPE 包含很强的标签信息。
- Reddit LapPE 包含中等标签信息，但明显弱于 arxiv。

## 4. LapPE 空间的类别聚集

### 4.1 全局成对距离

| dataset | distance | same mean | different mean | same-label AUC |
|---|---|---:|---:|---:|
| ogbn_arxiv | standardized Euclidean | 4.4156 | 4.6620 | 0.5386 |
| ogbn_arxiv | standardized cosine | 0.9161 | 1.0666 | 0.6272 |
| reddit | standardized Euclidean | 4.9330 | 4.6789 | 0.4761 |
| reddit | standardized cosine | 0.9611 | 1.0187 | 0.5378 |

观察：

- arxiv 同类样本在 cosine 空间中明显更近，但全局分离不是非常强。
- Reddit 全局类别聚集很弱。
- Reddit 的 Euclidean 距离甚至出现同类平均距离更大的现象。

### 4.2 LapPE kNN 标签一致性

| dataset | k=5 | k=10 | k=20 | chance |
|---|---:|---:|---:|---:|
| ogbn_arxiv | 0.7955 | 0.7903 | 0.7820 | 0.5004 |
| reddit | 0.6776 | 0.6717 | 0.6654 | 0.5415 |

结论：

- arxiv LapPE 有较强的局部类别结构。
- Reddit LapPE 只有中等局部类别结构。
- LapPE 更像局部结构信号，不是稳定的全局类别坐标系。

## 5. coarse graph 健康度

top-k=20 后：

| dataset | components | largest component | isolated | edge homophily | weighted homophily |
|---|---:|---:|---:|---:|---:|
| ogbn_arxiv | 1 | 1.0000 | 0 | 0.7579 | 0.8352 |
| reddit | 1 | 1.0000 | 0 | 0.6635 | 0.6602 |

两张图均满足：

- 只有一个连通分量；
- 最大连通分量覆盖全部节点；
- 没有孤立节点；
- 近零特征值数量为理论上的一个。

因此，top-k 没有把 coarse graph 直接切碎。

但“保持连通”不等于“谱结构健康”。Reddit top-k=20 的 spectral gap 只有
`0.00142`，未剪枝图为 `0.04219`。这表示图虽然连通，但已经形成多个弱连接区域。

## 6. top-k 对谱结构的影响

top-k=20 相对未剪枝图：

| dataset | edge retention | eigenvalue change | eigenspace similarity | distance Spearman |
|---|---:|---:|---:|---:|
| ogbn_arxiv | 0.1900 | 0.3498 | 0.8110 | 0.7939 |
| reddit | 0.0218 | 0.6945 | 0.6845 | 0.5703 |

### arxiv

- top-k=20 只保留约 `19.0%` 的边。
- 谱结构发生中等变化。
- LapPE 几何仍保留较多，但不是无损剪枝。
- top-k=40 时 distance Spearman 为 `0.8604`。
- top-k=80 时 distance Spearman 为 `0.9263`。

### Reddit

- top-k=20 只保留约 `2.18%` 的边。
- 特征值相对变化达到 `0.6945`。
- LapPE 成对距离相关性只有 `0.5703`。
- 即使 top-k=80，distance Spearman 也只有 `0.7508`。

结论：

- arxiv 的 top-k=20 存在中等谱失真。
- Reddit 的 top-k=20 存在严重谱失真。
- 固定绝对 top-k 不适合密度差异很大的 coarse graph。

## 7. LapPE 与 z_mil 的对比

### 7.1 arxiv 10 折

| representation | ACC mean ± std | F1 mean ± std | AUC mean ± std |
|---|---:|---:|---:|
| LapPE | 0.8393 ± 0.0000 | 0.8329 ± 0.0000 | 0.9202 ± 0.0000 |
| z_mil | 0.8635 ± 0.0151 | 0.8581 ± 0.0157 | 0.9386 ± 0.0130 |
| z_mil + LapPE | 0.8639 ± 0.0151 | 0.8584 ± 0.0160 | 0.9388 ± 0.0123 |

融合相对 `z_mil`：

| metric | mean delta | 折间表现 |
|---|---:|---|
| ACC | +0.00042 | 5 折提升，5 折下降或持平 |
| F1 | +0.00033 | 6 折提升，4 折下降 |
| AUC | +0.00022 | 6 折提升，4 折下降 |

该变化远小于折间标准差，不构成稳定增益。

邻域质量：

| representation | k=5 | k=10 | k=20 |
|---|---:|---:|---:|
| LapPE | 0.7955 | 0.7903 | 0.7820 |
| z_mil | 0.8338 ± 0.0225 | 0.8312 ± 0.0235 | 0.8272 ± 0.0237 |

arxiv 上 LapPE 有效，但 `z_mil` 更强。

### 7.2 Reddit 4 折

| representation | ACC mean ± std | F1 mean ± std | AUC mean ± std |
|---|---:|---:|---:|
| LapPE | 0.6823 ± 0.0000 | 0.6269 ± 0.0000 | 0.7289 ± 0.0000 |
| z_mil | 0.9183 ± 0.0184 | 0.8870 ± 0.0253 | 0.9709 ± 0.0091 |
| z_mil + LapPE | 0.9174 ± 0.0184 | 0.8859 ± 0.0250 | 0.9703 ± 0.0089 |

融合相对 `z_mil`：

- ACC：`-0.00088`；
- F1：`-0.00114`；
- AUC：`-0.00057`。

AUC 在 4 折中有 3 折下降，仅 1 折增加 `0.00015`。

邻域质量：

| representation | k=5 | k=10 | k=20 |
|---|---:|---:|---:|
| LapPE | 0.6776 | 0.6717 | 0.6654 |
| z_mil | 0.9086 ± 0.0218 | 0.9064 ± 0.0221 | 0.9032 ± 0.0223 |

Reddit 的关系结构主要存在于 learned representation space，而不是 LapPE 空间。

## 8. 轻量融合器是否使用 LapPE

这里的融合器是分别标准化 `z_mil` 和 LapPE 后训练的 Logistic Regression probe，
不是原端到端分类器。

### arxiv

| diagnostic | mean ± std |
|---|---:|
| zero LapPE AUC drop | 0.00659 ± 0.00504 |
| shuffle LapPE AUC drop | 0.02215 ± 0.01308 |
| zero LapPE ACC drop | 0.01447 ± 0.00567 |
| shuffle LapPE ACC drop | 0.03288 ± 0.01269 |
| position/z coefficient L2 ratio | 0.2301 ± 0.0371 |

arxiv 融合器确实使用 LapPE。但是单独重训 `z_mil` 几乎达到相同性能，说明 LapPE
提供的信息可以被 `z_mil` 替代。

### Reddit

| diagnostic | mean ± std |
|---|---:|
| zero LapPE AUC drop | 0.00011 ± 0.00049 |
| shuffle LapPE AUC drop | 0.00189 ± 0.00099 |
| zero LapPE ACC drop | 0.00017 ± 0.00134 |
| shuffle LapPE ACC drop | 0.00354 ± 0.00112 |
| position/z coefficient L2 ratio | 0.1243 ± 0.0028 |

Reddit 融合器基本不依赖 LapPE。

重要区别：

- “融合模型使用 LapPE”不代表“LapPE 提供不可替代的新信息”。
- arxiv 属于被使用但信息冗余。
- Reddit 属于信息较弱且基本被忽略。

## 9. 特征尺度

### arxiv

- `z_mil` 为 256 维，平均行 L2 范数约 `3.269`。
- LapPE 为 16 维，平均行 L2 范数约 `0.0655`。
- `z_mil` 行范数约为 LapPE 的 40 至 66 倍。
- `z_mil` 每维标准差约为 LapPE 的 3.6 至 5.6 倍。

### Reddit

- `z_mil` 平均行 L2 范数约 `22.50`。
- LapPE 平均行 L2 范数约 `0.0493`。
- 两者原始尺度差异更大。

轻量 probe 已分别标准化两个特征块，因此 probe 的无增益不能归因于尺度问题。
原端到端模型是否受尺度影响仍缺少 checkpoint 证据。

## 10. 对六个核心问题的回答

### 1. LapPE 是否真的包含标签相关信息？

是。

- arxiv：强。
- Reddit：中等。

### 2. coarse graph 是否提供有意义的位置结构？

- arxiv：提供有意义结构。
- Reddit：有一定结构，但明显弱于 `z_mil` 空间。

### 3. top-k 是否破坏 LapPE 谱结构？

是。

- arxiv：中等破坏。
- Reddit：严重破坏。

### 4. 模型是否真正利用位置编码？

- 轻量融合 probe：arxiv 会使用，Reddit 基本忽略。
- 原端到端模型：没有 checkpoint，无法严格确认。

### 5. 问题出在位置编码还是融合阶段？

- arxiv：主要是 LapPE 与 `z_mil` 信息冗余，不是 LapPE 无信息。
- Reddit：位置结构本身较弱，top-k 又造成严重谱失真；融合器也基本不使用它。

### 6. 当前任务是否缺少“位置”信息？

现有证据不能支持“任务缺少一个全局谱位置坐标”这一假设。

更准确的判断是：

- 子图之间确实存在关系信息；
- 关系信息主要体现在 `z_mil` 相似性和预测一致性中；
- 当前 LapPE 不是最合适的关系表示方式。

## 11. 最终建模判断

### LapPE 是否继续使用

`ogbn_arxiv`：

- 可以保留 LapPE-only probe，证明 coarse graph 含标签结构。
- 可以作为辅助消融。
- 不建议作为主要性能贡献，因为增量 AUC 只有约 `0.00022`。

`reddit`：

- 不建议继续作为主模型组件。
- 可以作为负结果和谱诊断证据。
- 不建议继续调 `pos_dim`、PositionMLP 深度或简单 concat。

总体上，应停止当前：

```text
coarse graph
  -> fixed top-k
  -> LapPE
  -> concat(z_mil, pos_emb)
```

作为主路线的继续调参。

## 12. 下一步如何学习子图关系

已有证据：

- arxiv `z_mil` kNN@16 同标签率约 `0.83`，原 coarse graph 约 `0.77`。
- Reddit `z_mil` kNN@16 同标签率约 `0.91`，原 coarse graph 约 `0.67`。
- Reddit coarse edge weight 对 same-label 的预测 AUC 约 `0.47`。
- `z_mil` cosine 对 Reddit same-label 的预测 AUC 约 `0.81`。
- 预测概率一致性对 same-label 的预测 AUC 可达到约 `0.95`。

推荐路线：

```text
子图编码器
  -> z_i = z_mil

候选关系图
  -> z_mil cosine kNN，优先 k=8 或 k=16
  -> 可选与原 coarse graph 取并集

边可靠性模型
  -> q_ij = MLP([
       z_i,
       z_j,
       |z_i-z_j|,
       z_i*z_j,
       cosine(z_i,z_j),
       coarse_weight_ij
     ])

边级消息聚合
  -> alpha_ij = softmax_j(q_ij)
  -> m_i = sum_j alpha_ij * V(z_j)

保守残差融合
  -> h_i = z_i + beta_i * m_i
  -> y_i = Classifier(h_i)
```

训练目标：

```text
L = L_node + lambda_edge * L_edge
```

其中：

```text
target_ij = 1[y_i == y_j]
L_edge = BCE(q_ij, target_ij)
```

只允许使用训练节点之间的标签构造 `L_edge`。验证和测试节点不能参与监督边标签。

建议初始设置：

```text
k = 16
lambda_edge = 0.1
beta initial value = 0.1
positive:negative edge sampling = 1:1 to 1:3
```

核心原则：

- 先判断每条边是否可信，再聚合。
- 保留强 `z_mil` 自身表示。
- 使用弱残差关系修正，不做强制平滑。
- 不再使用“先平均所有邻居，再做 node-level gate”的方式。

建议对照：

1. `MLP(z_mil)`；
2. `SAGE(z_mil kNN)`；
3. `edge-gated residual(z_mil kNN)`；
4. `edge-gated residual(kNN union coarse)`；
5. 去掉 `L_edge`；
6. 去掉 `coarse_weight`；
7. 不同 `k`：8、16、32。

重点指标：

- 相对 `MLP(z_mil)` 改对多少错误样本；
- 又破坏多少原本正确样本；
- edge scorer 的 same-label AUC；
- 筛选后边同质性；
- 消息残差 `beta` 分布；
- AUC、ACC、F1 和 loss 的折间稳定性。

## 13. 诊断脚本运行方式

基础诊断：

```bash
python lappe_diagnostics.py \
  --processed_data_dir /data/yg/Subgraph-MIL/Data/processed_data \
  --data_name_set ogbn_arxiv reddit \
  --lap_pe_dim 16 \
  --coarse_topk 20 \
  --topk_list 5,10,20,40,80 \
  --output_dir result/lappe_diagnostics
```

加入某一折 `z_mil`：

```bash
python lappe_diagnostics.py \
  --processed_data_dir /data/yg/Subgraph-MIL/Data/processed_data \
  --data_name_set ogbn_arxiv \
  --z_mil_path 'result/coarse_relation/ogbn_arxiv_zmil_fold0.pt' \
  --output_dir result/lappe_diagnostics_fold0
```

## 14. 当前局限

- Reddit 只有 4 折 `z_mil` 补充结果，不是完整 10 折。
- 没有保存原 LapPE 端到端模型 checkpoint。
- 因此不能对原模型执行严格的 zero/shuffle LapPE 消融。
- 轻量融合 probe 的结果只能证明可利用性和增量信息，不能完全代替端到端模型分析。

这些局限不影响当前主判断：

> LapPE 含有标签信息，但没有提供稳定的新增预测价值。子图关系学习应从固定谱位置
> 编码转向基于 `z_mil` 的候选图构建和边级可靠性学习。
