# coding=utf-8

import os
import random

import numpy as np
import torch


def set_seed(seed, cuda_deterministic=False):
  seed = int(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
  os.environ['PYTHONHASHSEED'] = str(seed)


def worker_init_fn(worker_id):
  worker_seed = (torch.initial_seed() + int(worker_id)) % (2 ** 32)
  random.seed(worker_seed)
  np.random.seed(worker_seed)
