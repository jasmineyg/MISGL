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
    if isinstance(model_output, dict) and 'ypred_A' in model_output:
        logits_A = model_output['ypred_A']
        bce = F.binary_cross_entropy_with_logits(logits_A.view(-1), targets.view(-1).float())
        bb_cfg = getattr(hparams, 'branch_b', None)
        use_b = bool(bb_cfg and bb_cfg.get('use', False))
        lam = float(bb_cfg.get('lambda_attn', 0.0)) if use_b else 0.0
        eps = float(bb_cfg.get('attn_eps', 1e-6)) if use_b else 1e-6

        # 注意力正则
        if use_b and 'branch_b' in model_output and lam > 0.0:
            a_pad = model_output['branch_b'].get('a_pad', None)
            mask_valid = model_output['branch_b'].get('mask_valid', None)
            if isinstance(a_pad, torch.Tensor) and isinstance(mask_valid, torch.Tensor):
                mask = mask_valid.float()
                a_masked = a_pad * mask
                N = mask_valid.sum(dim=1).float()
                denom = torch.log(torch.clamp(N, min=2.0))
                H = -(a_masked * torch.log(torch.clamp(a_pad, min=1e-12) + eps)).sum(dim=1)
                H_hat = H / denom
                y = targets.view(-1).float()
                L_attn = (y * H_hat + (1.0 - y) * (1.0 - H_hat)).mean()
                return bce + lam * L_attn
        return bce
    else:
        logits_A = model_output
        return F.binary_cross_entropy_with_logits(logits_A.view(-1), targets.view(-1).float())
