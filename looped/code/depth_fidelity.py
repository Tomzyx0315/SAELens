"""Per-depth faithfulness of the cross-loop SAE (A1).

The single-edit splice test (splice_eval.py) has no power at intermediate loops because later loops overwrite the
state. Instead, for each depth l we run Huginn for exactly l loops and replace the *final* (loop-l) state before it
is read out by Huginn's own coda + head (the model's real output at depth l). Comparisons at the same depth:

- CE_clean(l): unmodified depth-l output
- CE_sae(l): loop-l state replaced by sae.decode(sae.encode(.))
- CE_mean(l): loop-l state replaced by its per-sequence token mean (ablation baseline)
- CE_zero(l): replaced by zeros (the coda's ln_f normalizes; second ablation baseline)
- KL(clean || sae) and KL(clean || mean) of the next-token distributions
- loss recovered = (CE_mean - CE_sae) / (CE_mean - CE_clean)

Next-token CE over all positions (math/gsm8k: answer tokens only), per-domain corpora, fixed init seed.
Usage: python depth_fidelity.py --ckpt <final> --device cuda:0 --out depth_fidelity
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sae_lens.constants import ACTIVATION_SCALER_CFG_FILENAME
from sae_lens.saes.sae import TrainingSAE
from sae_lens.training.activation_scaler import ActivationScaler

from models import get_adapter
from paths import DATA

DOMAINS = ["web", "code", "math", "gsm8k_full", "wiki"]
DEPTHS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--docs_per_domain", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    sae = TrainingSAE.load_from_disk(str(ckpt), device=args.device).eval()
    scaler = ActivationScaler()
    scaler.load(ckpt / ACTIVATION_SCALER_CFG_FILENAME)
    ad = get_adapter("huginn", dtype=torch.bfloat16, device=args.device)

    def recon(s):
        return scaler.unscale(sae.decode(sae.encode(scaler(s.float())))).to(s.dtype)

    keys = ["clean", "sae", "mean", "zero"]
    acc = {d: {k: [0.0, 0] for k in keys + ["kl_sae", "kl_mean", "fvu_num", "fvu_den"]} for d in DEPTHS}
    per_dom = {dom: {d: {"clean": [0.0, 0], "sae": [0.0, 0]} for d in DEPTHS} for dom in DOMAINS}

    for dom in DOMAINS:
        df = pd.read_parquet(DATA / f"domain_eval/{dom}.parquet")
        df = df.sample(min(args.docs_per_domain, len(df)), random_state=args.seed)
        for _, row in df.iterrows():
            ids = torch.tensor(list(row["input_ids"]), device=args.device).unsqueeze(0)
            start = int(row["answer_start"]) if row["answer_start"] >= 0 else 1
            tgt = ids[0, start:]
            with torch.no_grad():
                st = ad.run_states(ids, max(DEPTHS), seed=args.seed)  # (32, T, D); state after loop l = st[l-1]
                for d in DEPTHS:
                    s = st[d - 1]
                    variants = {"clean": s, "sae": recon(s), "mean": s.mean(0, keepdim=True).expand_as(s),
                                "zero": torch.zeros_like(s)}
                    logp = {}
                    for k, v in variants.items():
                        lg = ad._readout(v)[start - 1:-1].float()
                        logp[k] = F.log_softmax(lg, -1)
                        ce = F.nll_loss(logp[k], tgt, reduction="sum").item()
                        acc[d][k][0] += ce
                        acc[d][k][1] += tgt.numel()
                        if k in ("clean", "sae"):
                            per_dom[dom][d][k][0] += ce
                            per_dom[dom][d][k][1] += tgt.numel()
                    pc = logp["clean"].exp()
                    for k in ("sae", "mean"):
                        kl = (pc * (logp["clean"] - logp[k])).sum(-1).sum().item()
                        acc[d][f"kl_{k}"][0] += kl
                        acc[d][f"kl_{k}"][1] += tgt.numel()
                    x = s[1:].float()
                    xr = variants["sae"][1:].float()
                    acc[d]["fvu_num"][0] += ((xr - x) ** 2).sum().item()
                    acc[d]["fvu_den"][0] += ((x - x.mean(0)) ** 2).sum().item()
        print(f"{dom}: done", flush=True)

    res = {"depths": DEPTHS, "per_depth": {}, "per_domain_delta_ce": {}}
    for d in DEPTHS:
        m = {k: acc[d][k][0] / max(acc[d][k][1], 1) for k in keys + ["kl_sae", "kl_mean"]}
        rec = (m["mean"] - m["sae"]) / max(m["mean"] - m["clean"], 1e-6)
        res["per_depth"][str(d)] = {
            "ce_clean": round(m["clean"], 4), "ce_sae": round(m["sae"], 4), "ce_mean": round(m["mean"], 4),
            "ce_zero": round(m["zero"], 4), "delta_ce_sae": round(m["sae"] - m["clean"], 4),
            "kl_sae": round(m["kl_sae"], 4), "kl_mean": round(m["kl_mean"], 4), "loss_recovered": round(rec, 3),
            "fvu": round(acc[d]["fvu_num"][0] / acc[d]["fvu_den"][0], 4),
        }
    for dom in DOMAINS:
        res["per_domain_delta_ce"][dom] = {
            str(d): round(per_dom[dom][d]["sae"][0] / max(per_dom[dom][d]["sae"][1], 1)
                          - per_dom[dom][d]["clean"][0] / max(per_dom[dom][d]["clean"][1], 1), 4) for d in DEPTHS}
    out = Path(__file__).resolve().parent.parent / "results/sae" / f"{args.out}.json"
    json.dump(res, open(out, "w"), indent=1)
    print(f"{'depth':>5} {'CEclean':>8} {'CEsae':>8} {'CEmean':>8} {'dCE':>7} {'KLsae':>7} {'KLmean':>7} {'rec':>6} {'FVU':>6}")
    for d in DEPTHS:
        r = res["per_depth"][str(d)]
        print(f"{d:>5} {r['ce_clean']:8.3f} {r['ce_sae']:8.3f} {r['ce_mean']:8.3f} {r['delta_ce_sae']:7.3f} "
              f"{r['kl_sae']:7.3f} {r['kl_mean']:7.3f} {r['loss_recovered']:6.3f} {r['fvu']:6.3f}")
    print("saved", out)


if __name__ == "__main__":
    main()
