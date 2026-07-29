import torch
from einops import rearrange, repeat, reduce
import os
import numpy as np
import random

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism settings (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_gauss_samecov(y, m, sigma):
    diff = y - m
    sq = (diff * diff).sum(dim=-1)
    return -0.5 * sq / (sigma ** 2)

def log_banana_target(z, a = 0.1, shift_x=-1.0, shift_y=1.0):
    x, y = z[0], z[1]
    # Apply shift in the opposite direction
    x_orig = x - shift_x      # x_orig ≈ 0 when x ≈ -1
    y_orig = y - shift_y      # y_orig ≈ 0.1 when y ≈ 1
    u = x_orig
    v = y_orig + a * (x_orig**2 - 1)
    return -0.5 * (u**2 / 10 + v**2)


def log_banana_target_batch(z, a=0.1, shift_x=-1.0, shift_y=1.0):
    """
    z: (..., 2)
    returns: (...,)
    """
    x = z[..., 0]
    y = z[..., 1]

    x_orig = x - shift_x
    y_orig = y - shift_y

    u = x_orig
    v = y_orig + a * (x_orig**2 - 1)

    return -0.5 * (u**2 / 10 + v**2)

def log_std_gaussian_target(z):
    """
    Standard d-dim Gaussian: log N(z | 0, I_d) up to additive constant.

    z: shape (d,)
    returns: scalar
    """
    return -0.5 * (z * z).sum()


def log_std_gaussian_target_batch(z):
    """
    Standard d-dim Gaussian: log N(z | 0, I_d) up to additive constant.

    z: shape (..., d)
    returns: shape (...,)
    """
    return -0.5 * (z * z).sum(dim=-1)
