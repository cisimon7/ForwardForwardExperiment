import numpy as np
import torch as th
import random


def seed_devices(seed):
    np.random.seed(seed)
    th.manual_seed(seed)
    random.seed(seed)
    
    
def layer_norm(z, eps=1e-8):
    return z / (th.sqrt(th.mean(z ** 2, dim=-1, keepdim=True)) + eps)
    