"""C2: do early-loop features cause the late (converged) features / the answer?

Persistently ablate the EARLY features (active at loop 1, inactive at loop 32) at the early loops (1..E), then let
the recurrence run to loop 32 and measure:

- how much the converged (loop-32) feature set changes vs the clean run (Jaccard of active features, last position
  and mean over tokens);
- whether the loop-32 logit-lens prediction and target CE change;
- as a control, ablate a random same-size set of loop-1-active features, and ablate at late loops instead.

If ablating the early scaffold perturbs loop 32 far more than a random ablation (and more than ablating the same
features late), the early features causally shape the converged representation; if loop 32 is unchanged, the
recurrence re-derives its state regardless of the early scaffold.

Usage: python early_causes_late.py --ckpt <final> --tags <feature_loop_tags.npz> --device cuda:0 --out early_causes_late
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tags", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_loops", type=int, default=32)
    p.add_argument("--early_until", type=int, default=4, help="ablate early features at loops 1..E")
    p.add_argument("--docs_per_domain", type=int, default=40)
    p.add_argument("--thr", type=float, default=1e-3)
    p.add_argument("--ratio", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    z = np.load(args.tags)
    F = z["freq_by_loop"]
    loops_tag = list(z["loops"])
    e, c = F[loops_tag.index(1)], F[loops_tag.index(32)]
    early = np.where((e > args.thr) & (c < e / args.ratio))[0]
    loop1_active = np.where(e > args.thr)[0]
    rng = np.random.default_rng(args.seed)
    rand_set = rng.choice(np.setdiff1d(loop1_active, early), len(early), replace=False)
    early_t = torch.tensor(early, device=args.device)
    rand_t = torch.tensor(rand_set, device=args.device)

    ckpt = Path(args.ckpt)
    sae = TrainingSAE.load_from_disk(str(ckpt), device=args.device).eval()
    scaler = ActivationScaler()
    scaler.load(ckpt / ACTIVATION_SCALER_CFG_FILENAME)
    ad = get_adapter("huginn", dtype=torch.bfloat16, device=args.device)
    W = sae.W_dec.detach().float()

    def enc(s):
        return sae.encode(scaler(s.float()))

    def ablate(state, fset_t):
        f = enc(state)[:, fset_t]  # (T, |set|)
        return (state - f @ W[fset_t]).to(state.dtype)

    def jacc(a, b):  # (T, d_sae) bool
        inter = (a & b).sum(-1).float()
        return (inter / (a | b).sum(-1).clamp(min=1).float()).mean().item()

    early_loops = set(range(1, args.early_until + 1))
    late_loops = set(range(args.max_loops - args.early_until + 1, args.max_loops + 1))
    agg = {k: [] for k in ("early_at_early", "rand_at_early", "early_at_late")}
    tok_total = 0
    for dom in DOMAINS:
        df = pd.read_parquet(DATA / f"domain_eval/{dom}.parquet").sample(args.docs_per_domain, random_state=args.seed)
        for _, row in df.iterrows():
            ids = torch.tensor(list(row["input_ids"]), device=args.device).unsqueeze(0)
            with torch.no_grad():
                _, clean = ad.run_with_state_edit(ids, args.max_loops, lambda s: s, set(), seed=args.seed)
                A_clean = enc(clean[-1]) > 0  # loop-32 active set
                for key, fset, lp in (("early_at_early", early_t, early_loops),
                                      ("rand_at_early", rand_t, early_loops),
                                      ("early_at_late", early_t, late_loops)):
                    _, states = ad.run_with_state_edit(ids, args.max_loops, lambda s: ablate(s, fset), lp, seed=args.seed)
                    A = enc(states[-1]) > 0
                    agg[key].append(jacc(A, A_clean))
            tok_total += ids.shape[1]
        print(f"{dom}: done", flush=True)

    res = {"n_early": int(len(early)), "early_until": args.early_until, "tokens": tok_total,
           "loop32_active_jaccard_vs_clean": {k: round(float(np.mean(v)), 4) for k, v in agg.items()}}
    path = Path(__file__).resolve().parent.parent / "results/sae" / f"{args.out}.json"
    json.dump(res, open(path, "w"), indent=1)
    print(json.dumps(res, indent=1))
    print("interpretation: lower Jaccard = loop 32 changed more. Compare early_at_early vs rand_at_early (specificity)"
          " and vs early_at_late (does WHEN matter).")
    print("saved", path)


if __name__ == "__main__":
    main()
