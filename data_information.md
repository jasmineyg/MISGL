# 数据特性与粗图关系分析记录

本文档整理当前围绕 `ogbn_arxiv` 与 `reddit` 两个数据集已经观察到的数据特性、粗图性质、二阶段关系模型表现，以及后续建模判断。

## 实验背景

当前框架把每个子图视为一个粗图节点：

```text
子图内部 GAT / Branch-B
  -> 固定子图表示 z_mil
  -> 在子图级关系图上训练二阶段关系模型
```

已尝试的关系模型包括：

- `mlp`: 只使用固定 `z_mil`，不使用子图之间关系。
- `gcn`: 在关系图上做 GCN 平滑。
- `appnp`: 先用 MLP 预测，再做 Personalized PageRank 风格传播。
- `sage`: GraphSAGE 风格聚合邻居，保留自身表示并拼接邻居均值。
- `gated_sage`: 在 SAGE 基础上加入 node-level gate，学习每个子图听多少邻居。

诊断脚本进一步比较了：

- 原始 coarse graph。
- 基于 `z_mil` 的 kNN 图。
- 边权、`z_mil` cosine、预测概率一致性对“边两端同标签”的解释能力。
- 关系模型把 MLP 改对/改错的样本特征。
- `gated_sage` 的 gate 是否真的区分可信邻居。

## ogbn_arxiv

### 粗图同质性

`ogbn_arxiv` 的原始 coarse graph 本身有较强标签相关性：

| 指标 | 数值 |
|---|---:|
| edge_homophily | 0.7705 |
| weighted_edge_homophily | 0.8387 |
| directed edges | 31,874 |

边权有一定解释力，但不强：

| 边特征 | 预测 same-label AUC |
|---|---:|
| coarse_weight | 0.5956 |
| z_mil_cosine | 0.6646 |
| mlp_prob_agreement | 0.8458 |
| sage_prob_agreement | 0.8461 |
| gated_sage_prob_agreement | 0.8593 |

分桶趋势：

| 特征 | lowest bin same-label | highest bin same-label |
|---|---:|---:|
| coarse_weight | 0.6957 | 0.8597 |
| z_mil_cosine | 0.4974 | 0.8971 |
| mlp_prob_agreement | 0.2714 | 0.9826 |
| gated_sage_prob_agreement | 0.2728 | 0.9986 |

结论：`ogbn_arxiv` 的 coarse graph 有信息，但 `z_mil` 相似度和预测概率一致性更能判断边是否可靠。

### z_mil kNN 图

`z_mil` kNN 图比原始 coarse graph 更干净：

| k | z_mil kNN same-label | coarse top-k same-label | kNN/coarse overlap |
|---:|---:|---:|---:|
| 8 | 0.8321 | 0.7837 | 0.2295 |
| 16 | 0.8288 | 0.7705 | 0.2435 |
| 32 | 0.8240 | 0.7705 | 0.1703 |
| 64 | 0.8192 | 0.7705 | 0.1134 |

结论：`z_mil` 空间里的近邻比原始 coarse graph 近邻更同质，但二者不是同一批边。

### 节点级信号

在 val/test 上，单节点自身预测已经很强，邻居标签/邻居预测接近但没有超过自身预测：

| signal | val AUC | test AUC |
|---|---:|---:|
| mlp_prob | 0.9358 | 0.9359 |
| sage_prob | 0.9466 | 0.9410 |
| gated_sage_prob | 0.9456 | 0.9407 |
| mlp_weighted_neighbor_prob | 0.9254 | 0.9256 |
| weighted_neighbor_positive_label_ratio | 0.9231 | 0.9231 |

结论：arxiv 的邻居信息有用，但不是明显强于 `z_mil` 自身预测，因此传播模型提升空间有限。

### 二阶段模型表现

最近一轮包含 `gated_sage` 的 10 折结果：

| model | Acc | AUC | AP | F1 | Loss |
|---|---:|---:|---:|---:|---:|
| mlp | 0.8680 | 0.9420 | 0.9464 | 0.8580 | 0.3367 |
| gcn | 0.8620 | 0.9417 | 0.9469 | 0.8535 | 0.3246 |
| appnp | 0.8655 | 0.9441 | 0.9499 | 0.8578 | 0.3236 |
| sage | 0.8655 | 0.9429 | 0.9496 | 0.8583 | 0.3086 |
| gated_sage | 0.8605 | 0.9421 | 0.9494 | 0.8526 | 0.3116 |

结论：

- `mlp(z_mil)` 已经很强。
- `sage/appnp` 在 AUC/AP/loss 上有一些优势，但 Acc/F1 没有稳定超过 MLP。
- `gated_sage` 在 arxiv 上没有带来收益，甚至略低于普通 SAGE。

### gated_sage 诊断

`gated_sage` gate 分布：

| gate | split | mean | std |
|---|---|---:|---:|
| gate1 | val/test | ~0.56 | ~0.22 |
| gate2 | val/test | ~0.88 | ~0.06 |

MLP 改对/改错样本：

| group | count | gate1 mean | gate2 mean | neighbor_same_label_ratio |
|---|---:|---:|---:|---:|
| baseline_right_model_wrong | 103 | 0.4906 | 0.8336 | 0.4698 |
| baseline_wrong_model_right | 84 | 0.5214 | 0.8390 | 0.5970 |
| both_correct | 3374 | 0.5734 | 0.8904 | 0.8366 |
| both_wrong | 439 | 0.4757 | 0.8459 | 0.3686 |

结论：gate 有一定分化，但不是强判别式地过滤坏邻居。它更像 node-level 的整体邻居融合强度，而不是 edge-level 的可信边选择。

## reddit

### 粗图同质性

`reddit` 的原始 coarse graph 明显更噪：

| 指标 | 数值 |
|---|---:|
| edge_homophily | 0.6736 |
| weighted_edge_homophily | 0.6749 |
| directed edges | 64,000 |

边权几乎不能解释边是否同标签：

| 边特征 | 预测 same-label AUC |
|---|---:|
| coarse_weight | 0.4714 |
| z_mil_cosine | 0.8117 |
| mlp_prob_agreement | 0.9474 |
| sage_prob_agreement | 0.9495 |
| gated_sage_prob_agreement | 0.9509 |

分桶趋势：

| 特征 | lowest bin same-label | highest bin same-label |
|---|---:|---:|
| coarse_weight | 0.6956 | 0.6883 |
| z_mil_cosine | 0.1096 | 0.9656 |
| mlp_prob_agreement | 0.0195 | 0.9957 |
| gated_sage_prob_agreement | 0.0195 | 0.9959 |

结论：Reddit 上原始 coarse edge weight 基本不能筛出可信边。`z_mil` cosine 和预测概率一致性才是真正强的边可靠性信号。

### z_mil kNN 图

Reddit 上 `z_mil` kNN 图远比 coarse graph 干净：

| k | z_mil kNN same-label | coarse top-k same-label | kNN/coarse overlap |
|---:|---:|---:|---:|
| 8 | 0.9112 | 0.6778 | 0.0452 |
| 16 | 0.9081 | 0.6736 | 0.0713 |
| 32 | 0.9042 | 0.6736 | 0.0595 |
| 64 | 0.8994 | 0.6736 | 0.0475 |

结论：Reddit 上“子图之间的信息”主要不在原始 coarse graph 里，而在 learned representation space 里。原始 coarse graph 与 `z_mil` kNN 图重合极低。

### 节点级信号

Reddit 的邻居标签/邻居预测明显弱于自身预测：

| signal | val AUC | test AUC |
|---|---:|---:|
| mlp_prob | 0.9348 | 0.9233 |
| sage_prob | 0.9373 | 0.9285 |
| gated_sage_prob | 0.9363 | 0.9280 |
| mlp_weighted_neighbor_prob | 0.7763 | 0.7808 |
| weighted_neighbor_positive_label_ratio | 0.7745 | 0.7745 |

结论：在 Reddit 的原始 coarse graph 上做强传播会把弱邻居信号混入强 `z_mil` 自身信号，因此 GCN/APPNP 性能下降是合理的。

### 二阶段模型表现

最近一轮包含 `gated_sage` 的 10 折结果：

| model | Acc | AUC | AP | F1 | Loss |
|---|---:|---:|---:|---:|---:|
| mlp | 0.8590 | 0.9329 | 0.9051 | 0.8030 | 0.3947 |
| gcn | 0.8143 | 0.8806 | 0.8290 | 0.7127 | 0.4269 |
| appnp | 0.8532 | 0.9083 | 0.8776 | 0.7794 | 0.4040 |
| sage | 0.8660 | 0.9341 | 0.9073 | 0.8072 | 0.3826 |
| gated_sage | 0.8688 | 0.9334 | 0.9075 | 0.8113 | 0.3759 |

结论：

- GCN 明显伤性能。
- APPNP 也低于 MLP，说明强平滑不适合 Reddit 的原始 coarse graph。
- SAGE 和 gated_sage 略高于 MLP，但提升较小。
- gated_sage 比普通 SAGE 略好，但优势不足以单独作为核心方法。

### gated_sage 诊断

`gated_sage` gate 分布：

| gate | split | mean | std |
|---|---|---:|---:|
| gate1 | val/test | ~0.74 | ~0.17 |
| gate2 | val/test | ~0.76 | ~0.24 |

MLP 改对/改错样本：

| group | count | gate1 mean | gate2 mean | neighbor_same_label_ratio |
|---|---:|---:|---:|---:|
| baseline_right_model_wrong | 116 | 0.7335 | 0.6666 | 0.4337 |
| baseline_wrong_model_right | 185 | 0.7267 | 0.6697 | 0.6568 |
| both_correct | 6826 | 0.7454 | 0.7731 | 0.7064 |
| both_wrong | 873 | 0.7257 | 0.7088 | 0.4508 |

结论：gate 没有明显学会“邻居同类率低就少听、邻居同类率高就多听”。`baseline_right_model_wrong` 和 `baseline_wrong_model_right` 的 gate 非常接近，说明 node-level gate 不足以解决边噪声问题。

## 跨数据集结论

### 1. 子图之间确实有信息

两个数据集里，边两端预测概率一致性都能很好预测 same-label：

| 数据集 | best prob agreement AUC |
|---|---:|
| ogbn_arxiv | 0.8593 |
| reddit | 0.9509 |

这说明跨子图关系不是无效的。

### 2. 原始 coarse graph 不是最优关系载体

尤其是 Reddit：

```text
coarse graph same-label @16 = 0.6736
z_mil kNN same-label @16 = 0.9081
```

这说明 learned embedding space 的近邻比原始粗图邻居更可靠。

### 3. 强传播容易伤害强自身表示

`mlp(z_mil)` 已经很强。如果邻居信号质量低于自身预测，GCN/APPNP 这类强平滑会拖累性能。Reddit 上这一点非常明显。

### 4. SAGE 稳定但提升小

SAGE 保留自身表示并拼接邻居均值，因此比 GCN/APPNP 更稳。但它的提升幅度较小，不足以单独构成有力贡献。

### 5. node-level gate 不够

当前 `gated_sage` 是先平均所有邻居，再决定整体听多少邻居。诊断显示它没有可靠地区分改对和改错样本，因此下一步不应继续只调 node gate。

## 当前建模判断

不建议继续单纯在原始 coarse graph 上调 GCN/APPNP/SAGE。

优先方向应从“换传播模型”转为“换图/修图/筛边”：

1. **z_mil kNN graph**
   - 用固定 `z_mil` 在表示空间构建 kNN 图。
   - 诊断显示该图在两个数据集上同质性都高于原始 coarse graph。

2. **coarse graph + z_mil similarity filter**
   - 原始 coarse graph 作为候选边。
   - 用 `cosine(z_i, z_j)` 或预测概率一致性过滤/重加权。

3. **edge-level gate**
   - 不再先平均所有邻居。
   - 对每条边学习 `edge_gate_ij`：

```text
edge_gate_ij = MLP([z_i, z_j, |z_i - z_j|, cosine(z_i, z_j), coarse_weight_ij])
message_i = sum_j edge_gate_ij * w_ij * z_j
```

目前已新建 `position-knn` 分支，优先实现并测试第一条路线：`z_mil kNN graph`。
