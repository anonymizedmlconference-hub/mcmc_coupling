import torch
from .utils import *
from tqdm import tqdm

def gaussian_maximal_coupling_relection(x1, x_ref, z, sigma):
    y_ref = x_ref + sigma * z

    logs_ref = log_gauss_samecov(y_ref, x_ref, sigma)
    logs_1   = log_gauss_samecov(y_ref, x1,   sigma)

    logu = torch.rand((), device=y_ref.device, dtype=y_ref.dtype).log()

    # coalescence
    if logu <= (logs_1 - logs_ref):
        z_used = (y_ref - x1) / sigma    # <-- critical fix
        return y_ref.clone(), z_used

    # reflection otherwise
    diff = x1 - x_ref
    norm = diff.norm()
    if norm == 0:
        # actually unreachable because logs_1==logs_ref => always accept,
        # but keep it safe
        return y_ref.clone(), (y_ref - x1) / sigma

    e = diff / norm
    z_reflect = z - 2.0 * torch.dot(z, e) * e
    y1 = x1 + sigma * z_reflect
    return y1, z_reflect


def _log_iso_gauss_unnorm(y, mu, sigma):
    # log N(y | mu, sigma^2 I) up to an additive constant (the constant cancels)
    sigma2 = sigma * sigma
    return -0.5 * (y - mu).pow(2).sum(dim=-1) / sigma2

def gaussian_maximal_coupling(x1, x_ref, z, sigma, max_tries=10_000):
    """
    Maximal coupling of N(x_ref, sigma^2 I) and N(x1, sigma^2 I) WITHOUT reflection.

    Inputs are single vectors:
      x1, x_ref, z: (D,)
      sigma: scalar

    Returns:
      y1:     sample from N(x1, sigma^2 I) coupled with y_ref
      z_used: (y1 - x1)/sigma
    """
    device = x_ref.device
    dtype  = x_ref.dtype
    sigma  = torch.as_tensor(sigma, device=device, dtype=dtype)

    # 1) draw from reference
    y_ref = x_ref + sigma * z

    logp = _log_iso_gauss_unnorm(y_ref, x_ref, sigma)
    logq = _log_iso_gauss_unnorm(y_ref, x1,   sigma)

    # Coalescence with prob min(1, q(y_ref)/p(y_ref))
    logu = torch.rand((), device=device, dtype=dtype).log()
    if logu <= (logq - logp):
        y1 = y_ref
        return y1, (y1 - x1) / sigma

    # 2) otherwise sample Y ~ (q - min(p,q)) / (1 - overlap)
    # Rejection sampling from q with acceptance prob 1 - min(1, p/q).
    for _ in range(max_tries):
        z1 = torch.randn_like(z)
        y1 = x1 + sigma * z1

        logp_y = _log_iso_gauss_unnorm(y1, x_ref, sigma)
        logq_y = _log_iso_gauss_unnorm(y1, x1,   sigma)

        # min(1, p/q) = exp(min(0, logp-logq))
        log_min_1_p_over_q = torch.minimum(torch.zeros((), device=device, dtype=dtype),
                                           logp_y - logq_y)

        logv = torch.rand((), device=device, dtype=dtype).log()

        # accept if v > min(1, p/q)
        if logv > log_min_1_p_over_q:
            return y1, z1

    raise RuntimeError("gaussian_maximal_coupling_no_reflection: max_tries exceeded")


def mh_gaussian(x0, log_target_func, sigma, *, len_gen=2**14):
    x_c = x0
    list_z, list_u = [], []
    list_x = [x_c.clone()]

    for _ in range(len_gen):
        z = torch.randn_like(x0)
        u = torch.rand((), device=x0.device, dtype=x0.dtype)
        x_p = x_c + z * sigma

        log_alpha = log_target_func(x_p) - log_target_func(x_c)
        if u.log() < log_alpha:
            x_c = x_p

        list_x.append(x_c.clone())
        list_z.append(z)
        list_u.append(u)

    return torch.stack(list_x), torch.stack(list_z), torch.stack(list_u)


def maxcoupl_2step(x0, *, ref_chain, z_ref, u_ref, log_target_func, sigma=1.0):
    len_gen = len(z_ref)
    x_c = x0
    list_z = torch.ones_like(z_ref) * torch.inf #[]
    list_x = torch.ones_like(ref_chain) * torch.inf #[x_c.clone()]
    list_x[0] = x_c.clone()

    for i in range(len_gen):
        z = z_ref[i]
        x_ref = ref_chain[i]
        u = u_ref[i]

        x_p, z_ = gaussian_maximal_coupling(x_c, x_ref, z, sigma=sigma)
        log_alpha = log_target_func(x_p) - log_target_func(x_c)

        if u.log() < log_alpha:
            x_c = x_p

        list_x[i+1] = x_c.clone() #list_x.append(x_c.clone())
        list_z[i] = z_ #list_z.append(z_)
        #list_u.append(u)
        if (x_c - ref_chain[i+1]).abs().sum() < 1e-8:
            #import pdb; pdb.set_trace()
            list_x[i+2:] = ref_chain[i+2:].clone()
            list_z[i+1:] = z_ref[i+1:]
            break
    assert torch.isfinite(list_x).all().item()
    assert torch.isfinite(list_z).all().item()
    #import pdb; pdb.set_trace()
    return list_x, list_z, u_ref
    #torch.stack(list_x), torch.stack(list_z), u_ref

@torch.no_grad()
def grand_maxcoupling_2steps(
    init_xs, num_chains=2**10, log_target_func=log_banana_target, sigma=1.0, tol=1e-8, mode='sequential'
):
    #if log_target_func == log_banana_target:
    #    d = 2
    if mode == 'sequential':
        ref_chain_idx = -1
    elif mode == 'star':
        ref_chain_idx = 0
    #init_xs = torch.randn((num_chains, d))
    #assert init_xs.shape == (num_chains, d)

    all_list_x = []
    all_list_z = []
    all_list_u = []

    # run the first chain
    list_x, list_z, list_u = mh_gaussian(init_xs[0], log_target_func, sigma)
    all_list_x.append(list_x)
    all_list_z.append(list_z)
    all_list_u = list_u  # share the same uniforms

    # couple the rest
    for i in tqdm(range(1, num_chains)):
        list_x_new, list_z_new, _ = maxcoupl_2step(
            init_xs[i],
            ref_chain=all_list_x[ref_chain_idx],
            z_ref=all_list_z[ref_chain_idx],
            u_ref=list_u,
            log_target_func=log_target_func,
            sigma=sigma,
        )

        all_list_x.append(list_x_new)
        all_list_z.append(list_z_new)


    all_list_x = torch.stack(all_list_x)

    meet = ((all_list_x.min(dim=0).values - all_list_x.max(dim=0).values).abs() < tol).prod(dim=-1)
    meet_index = (meet == 1).nonzero(as_tuple=True)[0][0].item()

    return all_list_x, meet_index

@torch.no_grad()
def grand_maxcoupling_1step(
    init_xs, num_chains=2**10, log_target_func=log_banana_target, sigma=1.0, tol=1e-8, mode='sequential'
):
    #if log_target_func == log_banana_target:
    #    d = 2
    if mode == 'sequential':
        ref_chain_idx = -1
    elif mode == 'star':
        ref_chain_idx = 0
    #init_xs = torch.randn((num_chains, d))
    #assert init_xs.shape == (num_chains, d)

    all_list_x = []
    all_list_z = []
    all_list_u = []

    # run the first chain
    list_x, list_z, list_u = mh_gaussian(init_xs[0], log_target_func, sigma)
    all_list_x.append(list_x)
    all_list_z.append(list_z)
    all_list_u = list_u  # share the same uniforms

    # couple the rest
    for i in tqdm(range(1, num_chains)):
        list_x_new, list_z_new, _ = maxcoupl_1step(
            init_xs[i],
            ref_chain=all_list_x[ref_chain_idx],
            z_ref=all_list_z[ref_chain_idx],
            u_ref=list_u,
            log_target_func=log_target_func,
            sigma=sigma,
        )

        all_list_x.append(list_x_new)
        all_list_z.append(list_z_new)


    all_list_x = torch.stack(all_list_x)

    #for i in range(len(all_list_x)):
    #    meet = ((all_list_x[i:i+2].min(dim=0).values - all_list_x[i:i+2].max(dim=0).values).abs() < tol).prod(dim=-1)
    #    meet_index = (meet == 1).nonzero(as_tuple=True)[0][0].item()
    #    print (meet_index)

    meet = ((all_list_x.min(dim=0).values - all_list_x.max(dim=0).values).abs() < tol).prod(dim=-1)
    meet_index = (meet == 1).nonzero(as_tuple=True)[0][0].item()

    return all_list_x, meet_index

def maxcoupl_1step(x0, *, ref_chain, z_ref, u_ref, log_target_func, sigma=1.0):
    len_gen = len(z_ref)
    x_c = x0
    list_z = torch.ones_like(z_ref) * torch.inf #[]
    list_x = torch.ones_like(ref_chain) * torch.inf #[x_c.clone()]
    list_x[0] = x_c.clone()

    for i in range(len_gen):
        z = z_ref[i]
        x_ref = ref_chain[i]
        x_ref_n = ref_chain[i+1]
        u = u_ref[i]

        x_c = gaussianMH_maximal_coupling(x_c, x_ref,x_ref_n, log_target_func, sigma=sigma)

        list_x[i+1] = x_c.clone()

        if (x_c - ref_chain[i+1]).abs().sum() < 1e-8:
            #import pdb; pdb.set_trace()
            list_x[i+2:] = ref_chain[i+2:].clone()
            break
    assert torch.isfinite(list_x).all().item()

    #import pdb; pdb.set_trace()
    return list_x, None, u_ref

def gaussianMH_maximal_coupling(x1, x_ref_c, x_ref_n, log_target_func, sigma, max_tries=10_000):
    """
    Maximal coupling of N(x_ref, sigma^2 I) and N(x1, sigma^2 I) WITHOUT reflection.

    Inputs are single vectors:
      x1, x_ref_c, x_ref_n: (D,)
      sigma: scalar

    Returns:
      y1:     sample from N(x1, sigma^2 I) coupled with y_ref
    """
    device = x_ref_c.device
    dtype  = x_ref_c.dtype
    sigma  = torch.as_tensor(sigma, device=device, dtype=dtype)
    logu = torch.rand((), device=device, dtype=dtype).log()

    def compute_logfxy(x, y):
        logq = _log_iso_gauss_unnorm(y, x, sigma)  # log q(y|x)
        logratio = log_target_func(y) - log_target_func(x)  # symmetric MH ratio in log
        log_alpha = torch.minimum(logratio, torch.zeros((), device=logratio.device, dtype=logratio.dtype))
        return logq + log_alpha

    def draw_transition(y):
        z_ = torch.randn_like(y)
        y_ = y + sigma * z_
        logu_ = torch.rand((), device=device, dtype=dtype).log()

        if logu_ < log_target_func(y_) - log_target_func(y):
            atom = False
            return y_, atom
        atom = True
        return y, True

    # 1) ref_chain do not sample a point mass:
    if (x_ref_c - x_ref_n).abs().sum() > 1e-10 and logu <= compute_logfxy(x1, x_ref_n) - compute_logfxy(x_ref_c, x_ref_n):
        return x_ref_n.clone()

    for _ in range(max_tries):
        y_p, atom = draw_transition(x1)
        logv = torch.rand((), device=device, dtype=dtype).log()

        if atom: return y_p

        if logv > compute_logfxy(x_ref_c, y_p) - compute_logfxy(x1, y_p):
            return y_p


    raise RuntimeError("gaussianMH_maximal_coupling: max_tries exceeded")
