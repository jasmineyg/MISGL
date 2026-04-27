# coding=utf-8

import torch
import torch.nn.functional as F


def cross_entropy(prediction, reference):
    return F.cross_entropy(prediction, reference, reduction='mean')


def get_gamma(epoch, gamma_start=0.3, gamma_end=0.6, warmup_epochs=20):
    return gamma_start if epoch < warmup_epochs else gamma_end


def bce_from_probs(probs, targets):
    probs = torch.clamp(probs.view(-1), min=1e-8, max=1 - 1e-8)
    targets = targets.view(-1).float()
    return -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs)).mean()

def fused_loss(model_output, targets, epoch, hparams):
    if isinstance(model_output, dict) and 'ypred_A' in model_output:
        logits_A = model_output['ypred_A']
    else:
        logits_A = model_output
    return F.binary_cross_entropy_with_logits(logits_A.view(-1), targets.view(-1).float())
