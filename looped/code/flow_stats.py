"""Feature-flow statistics with baselines (A2).

For every document (per-domain corpora, ~200/domain) we encode the loop state at LOOPS with the shared SAE and
measure, per token, how the active feature set changes between consecutive sampled loops. Baselines tell apart
genuine turnover from threshold flicker:

- seed baseline: the same loop under a second random init state (how different is the "same" state?)
- noise baselines: the clean state at loop l plus isotropic Gaussian noise of relative norm eps, re-encoded
  (encoder flicker under a perturbation of known size; eps swept to calibrate turnover against state distance).
  Also records the actual relative state change between consecutive loops, so turnover can be compared with the
  noise curve at the same perturbation size.

Metrics per loop transition l -> l':
- binary stay rate: |A_l & A_l'| / |A_l| (fraction of active features that remain active)
- weighted turnover: 1 - sum_i min(a_i, a'_i) / sum_i max(a_i, a'_i) over feature activations (magnitude-aware)

Results aggregated per document, then mean and 95% bootstrap CI over documents.
Usage: python flow_stats.py --ckpt <final> --device cuda:0 --out flow_stats
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sae_lens.constants import ACTIVATION_SCALER_CFG_FILENAME
from sae_lens.saes.sae import TrainingSAE
from sae_lens.training.activation_scaler import ActivationScaler

from models import get_adapter
from paths import DATA

DOMAINS = ["web", "code", "math", "gsm8k_full", "wiki"]
LOOPS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


def stay_and_turnover(a, b):
    """a, b: (T, d) activations >= 0. Returns per-token binary stay rate and weighted turnover (numpy, (T,))."""
    A, B = a > 0, b > 0
    stay = (A & B).sum(1).float() / A.sum(1).clamp(min=1).float()
    wt = 1 - torch.minimum(a, b).sum(1) / torch.maximum(a, b).sum(1).clamp(min=1e-8)
    return stay.cpu().numpy(), wt.cpu().numpy()


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return [round(float(x.mean()), 4), round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--docs_per_domain", type=int, default=200)
    p.add_argument("--noise_eps", default="0.02,0.05,0.1,0.2,0.4", help="relative noise norms (comma-separated)")
    p.add_argument("--noise_loop", type=int, default=32, help="loop whose state is perturbed for the noise curve")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    sae = TrainingSAE.load_from_disk(str(ckpt), device=args.device).eval()
    scaler = ActivationScaler()
    scaler.load(ckpt / ACTIVATION_SCALER_CFG_FILENAME)
    ad = get_adapter("huginn", dtype=torch.bfloat16, device=args.device)

    def enc(s):
        return sae.encode(scaler(s.float()))

    eps_list = [float(e) for e in args.noise_eps.split(",")]
    trans = [(LOOPS[i], LOOPS[i + 1]) for i in range(len(LOOPS) - 1)]
    doc_stats = {f"{a}->{b}": {"stay": [], "wturn": [], "rel_dist": []} for a, b in trans}
    seed_stats = {str(l): {"stay": [], "wturn": [], "rel_dist": []} for l in LOOPS}
    noise_stats = {str(e): {"stay": [], "wturn": []} for e in eps_list}
    gen = torch.Generator(device=args.device).manual_seed(0)

    def rel_dist(x, y):
        return ((x.float() - y.float()).norm(dim=-1) / x.float().norm(dim=-1)).mean().item()

    for dom in DOMAINS:
        df = pd.read_parquet(DATA / f"domain_eval/{dom}.parquet")
        df = df.sample(min(args.docs_per_domain, len(df)), random_state=0)
        for di, (_, row) in enumerate(df.iterrows()):
            ids = torch.tensor(list(row["input_ids"]), device=args.device).unsqueeze(0)
            with torch.no_grad():
                s1 = ad.run_states(ids, max(LOOPS), seed=2 * di)[:, 1:]  # drop BOS
                s2 = ad.run_states(ids, max(LOOPS), seed=2 * di + 1)[:, 1:]
                f1 = {l: enc(s1[l - 1]) for l in LOOPS}
                for a, b in trans:
                    st, wt = stay_and_turnover(f1[a], f1[b])
                    doc_stats[f"{a}->{b}"]["stay"].append(st.mean())
                    doc_stats[f"{a}->{b}"]["wturn"].append(wt.mean())
                    doc_stats[f"{a}->{b}"]["rel_dist"].append(rel_dist(s1[a - 1], s1[b - 1]))
                for l in LOOPS:
                    st, wt = stay_and_turnover(f1[l], enc(s2[l - 1]))
                    seed_stats[str(l)]["stay"].append(st.mean())
                    seed_stats[str(l)]["wturn"].append(wt.mean())
                    seed_stats[str(l)]["rel_dist"].append(rel_dist(s1[l - 1], s2[l - 1]))
                x = s1[args.noise_loop - 1].float()
                for e in eps_list:
                    noise = torch.randn(x.shape, device=x.device, generator=gen)
                    noise = noise / noise.norm(dim=-1, keepdim=True) * x.norm(dim=-1, keepdim=True) * e
                    st, wt = stay_and_turnover(f1[args.noise_loop], enc(x + noise))
                    noise_stats[str(e)]["stay"].append(st.mean())
                    noise_stats[str(e)]["wturn"].append(wt.mean())
        print(f"{dom}: done", flush=True)

    res = {"n_docs": len(doc_stats[f"{trans[0][0]}->{trans[0][1]}"]["stay"]), "noise_loop": args.noise_loop,
           "transition": {k: {m: boot_ci(v[m]) for m in v} for k, v in doc_stats.items()},
           "seed_baseline": {k: {m: boot_ci(v[m]) for m in v} for k, v in seed_stats.items()},
           "noise_baseline_by_eps": {k: {m: boot_ci(v[m]) for m in v} for k, v in noise_stats.items()}}
    out = Path(__file__).resolve().parent.parent / "results/sae" / f"{args.out}.json"
    json.dump(res, open(out, "w"), indent=1)
    print(f"n_docs={res['n_docs']}  values: [mean, 95% CI low, high]")
    print("loop transition: stay | weighted turnover | relative state distance")
    for k, v in res["transition"].items():
        print(f"  {k:8s} {v['stay']} {v['wturn']} {v['rel_dist']}")
    print("same loop, other init seed: stay | wturn | rel dist")
    for k, v in res["seed_baseline"].items():
        print(f"  loop {k:>2}: {v['stay']} {v['wturn']} {v['rel_dist']}")
    print(f"loop-{args.noise_loop} state + isotropic noise of relative norm eps: stay | wturn")
    for k, v in res["noise_baseline_by_eps"].items():
        print(f"  eps {k:>5}: {v['stay']} {v['wturn']}")
    print("saved", out)


if __name__ == "__main__":
    main()
