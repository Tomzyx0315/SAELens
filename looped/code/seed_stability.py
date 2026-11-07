"""B3: SAE seed stability. Compare SAEs that differ only in parameter-init seed (and, for the main v2 run, in data
order and length) on:

1. features: Hungarian matching of decoder directions between each pair; fraction of features with a matched
   counterpart at cos > 0.7 (the usual "shared feature" criterion) and > 0.9;
2. early / late classes: tag each SAE by loop on the fixed eval cache (same vectors for all SAEs), define early /
   late / persistent exactly as in SAE.md, and check whether the class proportions and the *matched* features keep
   their class across seeds (class agreement among matched pairs, vs chance).

Usage: python seed_stability.py --ckpts <ckpt1>,<ckpt2>,<ckpt3> --names a,b,c --device cuda:0 --out seed_stability
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from sae_lens.constants import ACTIVATION_SCALER_CFG_FILENAME
from sae_lens.saes.sae import TrainingSAE
from sae_lens.training.activation_scaler import ActivationScaler


def load(ckpt, dev):
    sae = TrainingSAE.load_from_disk(str(ckpt), device=dev).eval()
    sc = ActivationScaler()
    p = Path(ckpt) / ACTIVATION_SCALER_CFG_FILENAME
    if p.exists():
        sc.load(p)
    return sae, sc


@torch.no_grad()
def tag(sae, sc, X, tags, loops, dev, chunk=8192):
    freq = np.zeros((len(loops), sae.cfg.d_sae))
    for i, l in enumerate(loops):
        idx = np.where(tags[:, 0] == l)[0]
        cnt = torch.zeros(sae.cfg.d_sae, device=dev)
        for c in np.array_split(idx, max(1, len(idx) // chunk)):
            cnt += (sae.encode(sc(X[c].to(dev, torch.float32))) > 0).float().sum(0)
        freq[i] = (cnt / len(idx)).cpu().numpy()
    return freq


def classes(freq, loops, thr=1e-3, ratio=5.0):
    e, c = freq[loops.index(1)], freq[loops.index(32)]
    lab = np.full(freq.shape[1], "other", dtype=object)
    lab[(e > thr) & (c > thr)] = "persistent"
    lab[(e > thr) & (c < e / ratio)] = "early"
    lab[(c > thr) & (e < c / ratio)] = "late"
    return lab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", required=True)
    p.add_argument("--names", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eval_cache", default="/data_fs/zhangyuxuan/looped/sae/eval_cache.pt")
    p.add_argument("--per_loop", type=int, default=60000)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    ckpts, names = args.ckpts.split(","), args.names.split(",")

    ev = torch.load(args.eval_cache)
    X, tags, loops = ev["X"], ev["tags"].numpy(), list(ev["loops"])
    rng = np.random.default_rng(0)
    sel = np.concatenate([rng.choice(np.where(tags[:, 0] == l)[0], args.per_loop, replace=False) for l in (1, 32)])
    Xs, Ts = X[sel], tags[sel]
    del ev, X

    W, lab = {}, {}
    res = {"names": names, "per_sae": {}, "pairs": {}}
    for ck, n in zip(ckpts, names):
        sae, sc = load(ck, args.device)
        W[n] = torch.nn.functional.normalize(sae.W_dec.detach().float(), dim=-1).cpu().numpy()
        freq = tag(sae, sc, Xs, Ts, [1, 32], args.device)
        full = np.zeros((len(loops), freq.shape[1]))
        full[loops.index(1)], full[loops.index(32)] = freq[0], freq[1]
        lab[n] = classes(full, loops)
        res["per_sae"][n] = {k: int((lab[n] == k).sum()) for k in ("early", "late", "persistent", "other")}
        print(n, res["per_sae"][n], flush=True)
        del sae

    for a, b in combinations(names, 2):
        S = W[a] @ W[b].T
        ri, ci = linear_sum_assignment(-S)
        cs = S[ri, ci]
        m7 = cs > 0.7
        agree = {}
        for k in ("early", "late", "persistent"):
            sel_a = (lab[a][ri] == k) & m7
            agree[k] = {"n_matched_in_class_a": int(sel_a.sum()),
                        "frac_same_class_in_b": round(float((lab[b][ci][sel_a] == k).mean()), 3) if sel_a.any() else None,
                        "chance(class share in b)": round(float((lab[b] == k).mean()), 3)}
        res["pairs"][f"{a}~{b}"] = {"mean_matched_cos": round(float(cs.mean()), 3),
                                    "frac_cos>0.7": round(float(m7.mean()), 3),
                                    "frac_cos>0.9": round(float((cs > 0.9).mean()), 3),
                                    "class_agreement_among_matched(cos>0.7)": agree}
        print(f"{a}~{b}: matched cos mean {cs.mean():.3f} >0.7 {m7.mean():.3f} >0.9 {(cs > 0.9).mean():.3f}", flush=True)
        for k, v in agree.items():
            print(f"   {k}: {v}", flush=True)
    out = Path(__file__).resolve().parent.parent / "results/sae" / f"{args.out}.json"
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
