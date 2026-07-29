import torch
from .utils import *
import numpy as np
from tqdm import tqdm
import math
from einops import rearrange

def sample_gaussian_mixture(states, sigma, B=None, device="cuda"):
    """
    states: (N, D)
    sigma:  scalar
    B:      number of samples

    returns: (B, D)
    """
    N, D = states.shape
    if B is None:
        B = N * 16

    # sample mixture indices
    idx = torch.randint(0, N, (B,), device=device)

    # gather means
    means = states[idx]                 # (B, D)

    # add isotropic Gaussian noise
    samples = means + sigma * torch.randn(B, D, device=device)

    return samples

def sample_gaussian_mixture_with_loglik(
    states,
    sigma,
    B=None,
    device=None,
):
    """
    states: (N, D)
    sigma:  scalar
    B:      number of samples
    device: torch.device or string (default: cuda if available)

    returns:
      samples:      (B, D)
      loglik:       (B,)     log M(samples)
      loglik_comp:  (B, N)   log N(samples | mu_i, sigma^2 I)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    states = states.to(device)
    dtype = states.dtype
    N, D = states.shape

    if B is None:
        B = N * 16

    sigma = torch.as_tensor(sigma, device=device, dtype=dtype)
    sigma2 = sigma * sigma

    # ---- sample mixture ----
    idx = torch.randint(0, N, (B,), device=device)
    means = states[idx]

    eps = torch.randn(B, D, device=device, dtype=dtype)
    samples = means + sigma * eps

    # ---- per-component log-likelihood ----
    # diff: (B, N, D)
    diff = samples[:, None, :] - states[None, :, :]

    quad = diff.pow(2).sum(dim=-1) / sigma2          # (B, N)
    loglik_comp = -0.5 * quad

    log_norm = -0.5 * D * (
        math.log(2 * math.pi) + torch.log(sigma2)
    )
    loglik_comp = loglik_comp + log_norm             # (B, N)

    # ---- mixture log-likelihood ----
    loglik = torch.logsumexp(loglik_comp, dim=1) - math.log(N)

    return samples, loglik, loglik_comp.T


def pmc_multimodal(list_states, sigma=1.0):
    N, D  = list_states.shape
    log_w_min = np.log(1/N)

    prev_max_arrival_time = 0

    all_scores = None
    proposals = None
    #goal: sample next states, coupling using PMC.
    while True:
        new_ppls, loglik, loglik_comp = sample_gaussian_mixture_with_loglik(list_states, sigma=sigma)
        n_ppls = new_ppls.shape[0]
        exps = -torch.log(torch.rand(n_ppls))
        exps = exps.cuda()

        t_ = prev_max_arrival_time + exps.cumsum(0)
        t_ = t_.cuda()

        prev_max_arrival_time = t_[-1]

        scores = t_.log() + loglik
        scores = (scores[None,:] - loglik_comp)
        bd_s = t_[-1].log() + log_w_min

        if all_scores is None:
            all_scores = scores
            proposals = new_ppls
        else:
            all_scores = torch.cat([all_scores, scores], dim=1)
            proposals = torch.cat([proposals, new_ppls], dim=0)

        min_score, min_idx = all_scores.min(-1)

        if (min_score < bd_s).prod():
            break

    return proposals[min_idx, :]

def mh_gaussian_pmc(x0, log_target_func, sigma, *, len_gen=2**12, fast_term=True):
    x_c = x0
    list_u = []
    list_x = [x_c.clone()]

    for _ in range(len_gen):
        u = torch.rand((), device=x0.device, dtype=x0.dtype)
        x_p = pmc_multimodal(x_c, sigma=sigma)
        log_alpha = log_target_func(x_p) - log_target_func(x_c)

        r = (u.log() < log_alpha).float()
        x_c = x_p * r[..., None] + (1-r[..., None]) * x_c

        list_x.append(x_c.clone())
        list_u.append(u)

        if fast_term:
            #import pdb; pdb.set_trace()
            tol = 1e-8
            meet = ((x_c.min(dim=0).values - x_c.max(dim=0).values).abs() < tol).prod()


            if meet: # _index is not None:
                return None, None, len(list_x)-1 #meet_index



    list_x = torch.stack(list_x)
    list_u = torch.stack(list_u)

    list_x = rearrange(list_x, 'T B D -> B T D')
    tol = 1e-8
    meet = ((list_x.min(dim=0).values - list_x.max(dim=0).values).abs() < tol).prod(dim=-1)
    hits = (meet == 1).nonzero(as_tuple=True)[0]
    meet_index = int(hits[0]) if len(hits) else None

    #meet_index = (meet == 1).nonzero(as_tuple=True)[0][0].item()

    return list_x, list_u, meet_index



def pmc_multimodal_1step(list_states, log_target_func, sigma=1.0):
    N, D  = list_states.shape
    log_w_min = np.log(1/(2*N))

    prev_max_arrival_time = 0

    all_scores = None
    proposals = None
    all_u = None
    #goal: sample next states, coupling using PMC.
    while True:
        new_ppls, loglik, loglik_comp = sample_gaussian_mixture_with_loglik(list_states, sigma=sigma)
        u_ = torch.randint(0,2, (new_ppls.shape[0],)).cuda()
        #corrector
        log_pi_ppls = log_target_func(new_ppls)
        log_pi_states = log_target_func(list_states)
        corrector = log_pi_ppls[None,:] - log_pi_states[:,None]
        corrector = torch.minimum(corrector, torch.tensor(0.0)).exp()
        corrector = corrector*u_ + (1-corrector)*(1-u_)
        corrector = corrector.log()
        ###############
        n_ppls = new_ppls.shape[0]
        exps = -torch.log(torch.rand(n_ppls))
        exps = exps.cuda()

        t_ = prev_max_arrival_time + exps.cumsum(0)
        t_ = t_.cuda()

        prev_max_arrival_time = t_[-1]

        scores = t_.log() + loglik
        scores = (scores[None,:] - loglik_comp)
        scores = scores - corrector

        bd_s = t_[-1].log() + log_w_min

        if all_scores is None:
            all_scores = scores
            proposals = new_ppls
            all_u = u_
        else:
            all_scores = torch.cat([all_scores, scores], dim=1)
            proposals = torch.cat([proposals, new_ppls], dim=0)
            all_u = torch.cat([all_u, u_], dim=0)

        min_score, min_idx = all_scores.min(-1)

        if (min_score < bd_s).prod():
            break
    return proposals[min_idx, :], all_u[min_idx]

def mh_gaussian_pmc_1step(x0, log_target_func, sigma, *, len_gen=2**12, fast_term=False):
    x_c = x0
    list_u = []
    list_x = [x_c.clone()]

    for _ in range(len_gen):
        u = torch.rand((), device=x0.device, dtype=x0.dtype)
        x_p , u= pmc_multimodal_1step(x_c, log_target_func, sigma=sigma)
        x_c = x_p * u[..., None] + (1-u[..., None]) * x_c

        list_x.append(x_c.clone())
        list_u.append(u)

        if fast_term:
            #import pdb; pdb.set_trace()
            tol = 1e-8
            meet = ((x_c.min(dim=0).values - x_c.max(dim=0).values).abs() < tol).prod()


            if meet: # _index is not None:
                return None, None, len(list_x)-1 #meet_index



    list_x = torch.stack(list_x)
    list_u = torch.stack(list_u)

    list_x = rearrange(list_x, 'T B D -> B T D')
    tol = 1e-8
    meet = ((list_x.min(dim=0).values - list_x.max(dim=0).values).abs() < tol).prod(dim=-1)
    hits = (meet == 1).nonzero(as_tuple=True)[0]
    meet_index = int(hits[0]) if len(hits) else None

    return list_x, list_u, meet_index
