import os
import numpy as np
import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# IMPORTANT: make sure these imports work inside subprocesses too
from src.max_coupling import *
from src.pmc_coupling import *
from src.utils import *
import argparse
import yaml
from pathlib import Path

from datetime import datetime


num_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
if num_cpus is not None:
    num_cpus = int(num_cpus)
else:
    num_cpus = os.cpu_count()

print("CPUs allocated:", num_cpus)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.yaml"
    )
    return parser.parse_args()


def _worker_run_star(worker_id, n_iters, n_chains, d, sigma, nu, seed):
    # Each process should not spawn extra torch threads (avoid oversubscription)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Make RNG independent across processes
    set_seed(seed + worker_id)
    g = torch.Generator().manual_seed(seed + 10_000 + worker_id)

    local1step = []
    local2steps = []

    with torch.inference_mode():
        for _ in range(n_iters):
            init_xs = torch.randn((n_chains, d), generator=g) #10 * (torch.rand((n_chains, d), generator=g) - 0.5)
            _, meet_index_star1step = grand_maxcoupling_1step(
                init_xs, num_chains=n_chains, log_target_func = log_std_cauchy_target , sigma=sigma, nu=nu, mode="star"
            )
            _, meet_index_star2steps = grand_maxcoupling_2steps(
                init_xs, num_chains=n_chains, log_target_func = log_std_cauchy_target , sigma=sigma, nu=nu, mode="star"
            )
            local1step.append(meet_index_star1step)
            local2steps.append(meet_index_star2steps)
            #print (data.mean(1), data.std(1))
            #import pdb; pdb.set_trace()
    return worker_id, local1step, local2steps


def run_star_10cores(total_iters, n_chains, d, sigma, nu, seed=42, n_workers=10):
    base, rem = divmod(total_iters, n_workers)
    chunk_sizes = [base + (1 if i < rem else 0) for i in range(n_workers)]

    # "spawn" is safest across environments (esp. notebooks)
    ctx = mp.get_context("spawn")

    results = [None] * n_workers
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futs = []
        for wid, n_iters in enumerate(chunk_sizes):
            if n_iters == 0:
                continue
            futs.append(ex.submit(_worker_run_star, wid, n_iters, n_chains, d, sigma, nu, seed))

        for fu in as_completed(futs):
            wid, local1step, local2steps = fu.result()
            results[wid] = (local1step, local2steps)

    # Deterministic stitch: worker 0 chunk, then worker 1 chunk, etc.
    all_meet_times_1step = []
    all_meet_times_2steps = []
    for item in results:
        if item is None:
            continue
        local1, local2 = item
        all_meet_times_1step.extend(local1)
        all_meet_times_2steps.extend(local2)

    return np.asarray(all_meet_times_1step), np.asarray(all_meet_times_2steps)


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    n_chains = cfg["n_chains"]
    set_seed(cfg["seed"])
    d = cfg["d"]
    sigma = (2.38/np.sqrt(d))
    nu = 2.0

    meet_1, meet_2 = run_star_10cores(
        total_iters=10000, 
        n_chains=n_chains,
        d=d,
        sigma=sigma,
        nu=nu,
        seed=cfg["seed"],
        n_workers=num_cpus,
    )
    mean_1 = meet_1.mean()
    mean_2 = meet_2.mean()
    var_1 = meet_1.var()
    var_2 = meet_2.var()
    print("mean 1-step:", meet_1.mean())
    print("mean 2-steps:", meet_2.mean())

    # ---- write results to text file ----
    config_path = Path(args.config)
    exp_name = cfg.get("name", config_path.stem)  # "name" from YAML if present, else filename stem
    out_dir = Path(cfg.get("out_dir", config_path.parent))
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = out_dir / f"{exp_name}_results_maximal.txt"

    with open(result_path, "w") as f:
        f.write(f"name: {exp_name}\n")
        f.write(f"config_path: {config_path.resolve()}\n")
        f.write(f"timestamp: {timestamp}\n\n")

        f.write("config:\n")
        f.write(yaml.safe_dump(cfg, sort_keys=True))
        f.write("\n")

        f.write("results:\n")
        f.write(f"Maximal Coupling (1step) on Gaussian MH: mean={mean_1:.6g}, var={var_1:.6g}\n")
        f.write(f"Maximal Coupling (2step) on Gaussian MH: mean={mean_2:.6g}, var={var_2:.6g}\n")

    print(f"Saved results to: {result_path}")

