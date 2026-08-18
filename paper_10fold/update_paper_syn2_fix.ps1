param(
    [Parameter(Mandatory = $true)]
    [string]$PaperPath
)

$ErrorActionPreference = 'Stop'
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$original = [System.IO.File]::ReadAllText($PaperPath, [System.Text.Encoding]::UTF8)
$updated = $original

$replacements = [ordered]@{
    '| GAT+mean pool | 0.7900 ± 0.0424 | 0.5230 ± 0.0492 | 0.7910 ± 0.0619 |' = '| GAT+mean pool | 0.7900 ± 0.0424 | 0.5380 ± 0.0461 | 0.7910 ± 0.0619 |'
    '| MIL-HEAD | 0.8990 ± 0.0328 | 0.5280 ± 0.0573 | 0.8780 ± 0.0352 |' = '| MIL-HEAD | 0.8990 ± 0.0328 | 0.5500 ± 0.0411 | 0.8780 ± 0.0352 |'
    '| POS-HEAD | 0.8860 ± 0.0337 | 0.5360 ± 0.0366 | 0.9740 ± 0.0143 |' = '| POS-HEAD | 0.8860 ± 0.0337 | 0.8830 ± 0.0350 | 0.9740 ± 0.0143 |'
    '| MISGL | **0.8990 ± 0.0412** | 0.5320 ± 0.0266 | **0.9940 ± 0.0084** |' = '| MISGL | **0.8990 ± 0.0412** | **0.8980 ± 0.0391** | **0.9940 ± 0.0084** |'
    'Syn1 中 MIL-HEAD 与 MISGL 的平均准确率并列最高（0.8990）；Syn3 中 MISGL 达到 0.9940，比 GAT+mean pool、MIL-HEAD 和 POS-HEAD 分别提高 20.30、11.60 和 2.00 个百分点。Syn2 的四种方法均接近随机水平，最高值为 POS-HEAD 的 0.5360，表明该弱正实例设置仍是当前模型的主要困难情形。' = 'Syn1 中 MIL-HEAD 与 MISGL 的平均准确率并列最高（0.8990）；Syn3 中 MISGL 达到 0.9940，比 GAT+mean pool、MIL-HEAD 和 POS-HEAD 分别提高 20.30、11.60 和 2.00 个百分点。Syn2 的 Stage-1 表征仍接近随机水平：MIL-HEAD 仅比 GAT+mean pool 提高 1.20 个百分点（0.5500 vs. 0.5380）；但冻结表征后，POS-HEAD 和 MISGL 分别达到 0.8830 和 0.8980，MISGL 相比 MIL-HEAD 提高 34.80 个百分点。这表明在弱正实例信号但粗图同质性较强的设置中，Stage-2 能够利用 bag 间关系恢复判别结构。'
    '| Syn2 | 500 | 0.9440 | 0.4409 | 0.9616 (300) | 0.9177 (200) |' = '| Syn2 | 500 | 1.0018 | 0.4936 | 0.9890 (258) | 1.0155 (242) |'
    '除 Syn2 外，其余 14 个数据集或划分版本均满足平均 $E_i>1$ 且平均 Ranking AUC $>0.5$。其中 Reddit 原始数据的 $E_i=7.6215$、AUC=0.9251，Syn1 和 Syn3 的 AUC 分别为 0.9981 和 0.9927，说明 MIL-HEAD 在这些设置中能够把注意力集中到真实正实例上。Syn2 的 $E_i=0.9440$、AUC=0.4409，与其接近随机的分类准确率一致，不能支持 attention 已识别关键正实例的解释。' = '除 Syn2 外，其余 14 个数据集或划分版本均同时满足平均 $E_i>1$ 且平均 Ranking AUC $>0.5$。其中 Reddit 原始数据的 $E_i=7.6215$、AUC=0.9251，Syn1 和 Syn3 的 AUC 分别为 0.9981 和 0.9927，说明 MIL-HEAD 在这些设置中能够把注意力集中到真实正实例上。Syn2 的 $E_i=1.0018$、AUC=0.4936，均接近随机注意力基线；这与其 Stage-1 分类准确率较低一致，同时说明 POS-HEAD 和 MISGL 的高准确率主要来自 Stage-2 对粗图关系的利用，而不是 MIL attention 已识别关键正实例。'
}

foreach ($pair in $replacements.GetEnumerator()) {
    $count = ([regex]::Matches($updated, [regex]::Escape($pair.Key))).Count
    if ($count -ne 1) {
        throw "Expected exactly one occurrence, found ${count}: $($pair.Key)"
    }
    $updated = $updated.Replace($pair.Key, $pair.Value)
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$backupPath = "$PaperPath.before_syn2_fix_$stamp.bak"
[System.IO.File]::WriteAllText($backupPath, $original, $utf8NoBom)
[System.IO.File]::WriteAllText($PaperPath, $updated, $utf8NoBom)

[pscustomobject]@{
    paper = $PaperPath
    backup = $backupPath
    replacement_count = $replacements.Count
    original_sha256 = (Get-FileHash -LiteralPath $backupPath -Algorithm SHA256).Hash
    updated_sha256 = (Get-FileHash -LiteralPath $PaperPath -Algorithm SHA256).Hash
}
