"""Splice the SAE reconstruction back into Huginn and measure its effect on the model.

The SAE is trained on the loop state (output of core_block[-1]). We can replace that state at chosen loops with the
SAE reconstruction and let the recurrence continue, which tests whether the SAE captures the information the model
actually uses, and how a per-loop reconstruction error propagates through later loops.

Metrics on the per-domain eval corpora (next-token CE over valid positions; for math/gsm8k only answer tokens):

1. CE_clean: unmodified model.
2. CE_replace_all: replace the loop state at every loop with sae.decode(sae.encode(state)).
3. CE_replace_one[l]: replace only at loop l (measures how much a single-loop edit survives to the output).
4. CE_meanabl[l]: replace the loop-l state with the batch-mean state (ablation baseline for "loss recovered").
5. Error propagation: inject the SAE reconstruction at loop l and record ||edited - clean|| of the state at every
   later loop, to see whether the recurrence amplifies or damps the reconstruction error.

Loss recovered at loop l = (CE_meanabl[l] - CE_replace_one[l]) / (CE_meanabl[l] - CE_clean).

Usage: python splice_eval.py --ckpt <final> --device cuda:0 --loops 32 --out <stem>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sae_lens.saes.sae import TrainingSAE
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.constants import ACTIVATION_SCALER_CFG_FILENAME

from models import get_adapter
from paths import DATA

DOMAINS = ["web", "code", "math", "gsm8k_full", "wiki"]
EDIT_LOOPS = [1, 2, 4, 8, 16, 32]


def ce(logits, ids, start):
    """Mean next-token CE over positions >= start (0-based target index)."""
    lg = logits[:-1].float()
    tgt = ids[0, 1:]
    if start > 0:
        lg, tgt = lg[start - 1:], tgt[start - 1:]
    return F.cross_entropy(lg, tgt).item(), tgt.numel()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--loops", type=int, default=32, help="number of recurrences to run")
    p.add_argument("--docs_per_domain", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    sae = TrainingSAE.load_from_disk(str(ckpt), device=args.device).eval()
    scaler = ActivationScaler()
    scaler.load(ckpt / ACTIVATION_SCALER_CFG_FILENAME)
    ad = get_adapter("huginn", dtype=torch.bfloat16, device=args.device)

    def recon(s):  # (T, D) -> SAE reconstruction in model space
        return scaler.unscale(sae.decode(sae.encode(scaler(s.float()))))

    edit_loops_present = [l for l in EDIT_LOOPS if l <= args.loops]
    agg = {k: [0.0, 0] for k in (["clean", "all"] + [f"one_{l}" for l in edit_loops_present]
                                 + [f"abl_{l}" for l in edit_loops_present])}
    prop = {l: np.zeros(args.loops) for l in edit_loops_present}  # ||edited-clean|| per later loop
    prop_n = {l: 0 for l in edit_loops_present}

    for dom in DOMAINS:
        df = pd.read_parquet(DATA / f"domain_eval/{dom}.parquet").sample(
            min(args.docs_per_domain, 10**9), random_state=args.seed)
        for _, row in df.iterrows():
            ids = torch.tensor(list(row["input_ids"]), device=args.device).unsqueeze(0)
            start = int(row["answer_start"]) if row["answer_start"] >= 0 else 1
            # clean
            logit_clean, states_clean = ad.run_with_state_edit(ids, args.loops, lambda s: s, set(), seed=args.seed)
            c, n = ce(logit_clean, ids, start)
            agg["clean"][0] += c * n
            agg["clean"][1] += n
            # replace all loops
            lg, _ = ad.run_with_state_edit(ids, args.loops, recon, set(range(1, args.loops + 1)), seed=args.seed)
            c, _ = ce(lg, ids, start)
            agg["all"][0] += c * n
            agg["all"][1] += n
            # per-loop single edit + mean ablation + error propagation
            for l in edit_loops_present:
                lg, states = ad.run_with_state_edit(ids, args.loops, recon, {l}, seed=args.seed)
                c, _ = ce(lg, ids, start)
                agg[f"one_{l}"][0] += c * n
                agg[f"one_{l}"][1] += n
                # error propagation: ||edited - clean|| at each loop >= l
                for j in range(l - 1, args.loops):
                    prop[l][j] += (states[j] - states_clean[j]).float().norm().item() / states_clean[j].float().norm().item()
                prop_n[l] += 1
                # mean-ablation baseline: replace loop-l state by its per-sequence mean over tokens
                mean_state = states_clean[l - 1].mean(0, keepdim=True).expand_as(states_clean[l - 1])
                lg, _ = ad.run_with_state_edit(ids, args.loops, lambda s, m=mean_state: m, {l}, seed=args.seed)
                c, _ = ce(lg, ids, start)
                agg[f"abl_{l}"][0] += c * n
                agg[f"abl_{l}"][1] += n
        print(f"{dom}: done", flush=True)

    out = {k: round(v[0] / max(v[1], 1), 4) for k, v in agg.items()}
    ce_clean = out["clean"]
    res = {
        "loops": args.loops,
        "ce_clean": ce_clean,
        "ce_replace_all_loops": out["all"],
        "delta_ce_replace_all": round(out["all"] - ce_clean, 4),
        "per_loop": {str(l): {
            "ce_replace_one": out[f"one_{l}"],
            "delta_ce": round(out[f"one_{l}"] - ce_clean, 4),
            "ce_mean_ablation": out[f"abl_{l}"],
            "loss_recovered": round((out[f"abl_{l}"] - out[f"one_{l}"]) / max(out[f"abl_{l}"] - ce_clean, 1e-6), 3),
        } for l in edit_loops_present},
        "error_propagation_relnorm": {str(l): (prop[l] / max(prop_n[l], 1)).round(4).tolist()
                                      for l in edit_loops_present},
    }
    json.dump(res, open(Path("../results/sae") / f"{args.out}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "error_propagation_relnorm"}, indent=1))
    print("\nerror propagation ||edit-clean||/||clean|| at each loop after injecting at loop l:")
    for l in edit_loops_present:
        v = res["error_propagation_relnorm"][str(l)]
        print(f"  inject@{l:2d}: " + " ".join(f"{x:.2f}" for x in v[l - 1:][:min(12, len(v))]))
    print("saved", Path("../results/sae") / f"{args.out}.json")


if __name__ == "__main__":
    main()
