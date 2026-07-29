from src.max_coupling import *
from src.pmc_coupling import *
from src.utils import *
import argparse
import yaml
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config.yaml"
    )
    return parser.parse_args()

args = parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

n_chains = cfg["n_chains"]
d = cfg["d"]
set_seed(cfg["seed"])

sigma = (2.4/np.sqrt(d))
nu=2.0
print ("Number of chains: ", n_chains, 'Dimensions: ', d)


print ("---------------------------------------------")
meeting_times_2step = []
meeting_times_1step = []
dist = torch.distributions.Cauchy(loc=0.0, scale=1.0)
device='cuda'
for i in tqdm(range(10000)): 
    init_xs = torch.randn((n_chains, d))
    _, _, meet_idx_pmc2step = mh_tstudent_pmc(init_xs.to(device), log_target_func=log_std_cauchy_target_batch, sigma=sigma, nu=nu, fast_term=True)
    _, _, meet_idx_pmc1step = mh_tstudent_pmc_1step(init_xs.to(device), log_target_func=log_std_cauchy_target_batch, sigma=sigma, nu=nu, fast_term=True)
    meeting_times_2step.append(meet_idx_pmc2step)
    meeting_times_1step.append(meet_idx_pmc1step)
    print (meet_idx_pmc2step, meet_idx_pmc1step)
    
meeting_times_2step = np.asarray(meeting_times_2step)
meeting_times_1step = np.asarray(meeting_times_1step)
print ("PMC Coupling (1steps) on t-student MH: ", meeting_times_1step.mean(), meeting_times_1step.var())
print ("PMC Coupling (2steps) on t-student MH: ", meeting_times_2step.mean(), meeting_times_2step.var())

mean_1 = float(meeting_times_1step.mean())
var_1  = float(meeting_times_1step.var())
mean_2 = float(meeting_times_2step.mean())
var_2  = float(meeting_times_2step.var())

# ---- write results to text file ----
config_path = Path(args.config)
exp_name = cfg.get("name", config_path.stem)  # "name" from YAML if present, else filename stem
out_dir = Path(cfg.get("out_dir", config_path.parent))
out_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_path = out_dir / f"{exp_name}_results_{timestamp}.txt"

with open(result_path, "w") as f:
    f.write(f"name: {exp_name}\n")
    f.write(f"config_path: {config_path.resolve()}\n")
    f.write(f"timestamp: {timestamp}\n\n")

    f.write("config:\n")
    f.write(yaml.safe_dump(cfg, sort_keys=True))
    f.write("\n")

    f.write("results:\n")
    f.write(f"PMC Coupling (1step) on Gaussian MH: mean={mean_1:.6g}, var={var_1:.6g}\n")
    f.write(f"PMC Coupling (2step) on Gaussian MH: mean={mean_2:.6g}, var={var_2:.6g}\n")

print(f"Saved results to: {result_path}")
