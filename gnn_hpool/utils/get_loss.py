# coding=utf-8

import torch
import torch.nn.functional as F


def cross_entropy(prediction, reference):
    return F.cross_entropy(prediction, reference, reduction='mean')


def get_gamma(epoch, gamma_start=0.3, gamma_end=0.6, warmup_epochs=20):
    return gamma_start if epoch < warmup_epochs else gamma_end


def bce_from_probs(probs, targets):
    probs = torch.clamp(probs.view(-1), min=1e-8, max=1-1e-8)
    targets = targets.view(-1).float()
    return -(targets*torch.log(probs) + (1-targets)*torch.log(1-probs)).mean()


def fused_loss(model_output, targets, epoch, hparams):
    # 处理两种输出格式：字典格式（有分支B）和张量格式（无分支B）
    if isinstance(model_output, dict) and 'ypred_A' in model_output:
        # 字典格式：有分支B的情况
        logits_A = model_output['ypred_A']  # [B] 或 [B,1]
        
        # 添加强约束断言
        if 'branch_b' in model_output and model_output['branch_b'] is not None:
            assert logits_A.size(0) == model_output['branch_b']['y_B'].size(0), \
                f"Batch size mismatch: logits_A {logits_A.shape} vs y_B {model_output['branch_b']['y_B'].shape}"
        
        use_b = ('branch_b' in model_output) and (model_output['branch_b'] is not None) \
                and ('y_B' in model_output['branch_b']) \
                and getattr(hparams, 'branch_b', None) and hparams.branch_b.get('use', False)

        if use_b:
            # 训练：在概率域融合
            p_A = torch.sigmoid(logits_A).view(-1)      # [B]
            p_B = model_output['branch_b']['y_B'].view(-1) # [B]
            
            assert p_A.shape == p_B.shape, f"Shape mismatch: p_A {p_A.shape} vs p_B {p_B.shape}"
            
            bb = hparams.branch_b
            gamma = get_gamma(epoch, bb.get('gamma_start',0.3), bb.get('gamma_end',0.6), bb.get('warmup_epochs',20))
            p_fused = gamma * p_B + (1 - gamma) * p_A
            return bce_from_probs(p_fused, targets)
        else:
            # A-only情况：使用BCEWithLogits
            return F.binary_cross_entropy_with_logits(logits_A.view(-1), targets.view(-1).float())
    else:
        # 张量格式：无分支B的情况（兼容旧版本）
        logits_A = model_output  # [B] 或 [B,1]
        return F.binary_cross_entropy_with_logits(logits_A.view(-1), targets.view(-1).float())
