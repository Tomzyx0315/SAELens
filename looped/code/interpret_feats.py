"""Interpret a given list of SAE features (e.g. the early / late-emerging sets) by collecting their max-activating
token contexts at a specified loop, across the per-domain eval corpora.

Unlike interpret_loops (which samples features), this targets an explicit feature set so we can read the semantics
of exactly the early (loop-1-only) and late-emerging (loop-32-only) features. Early features are interpreted at
loop 1, late features at loop 32 (each at the loop where they are active). Feature sets come from
feature_loop_tags.npz: early = freq@1 > thr & freq@32 < freq@1/ratio; late = freq@32 > thr & freq@1 < freq@32/ratio.

Output JSON: for each class, the feature list and, for a sample, the top contexts + per-domain firing share.

Usage: python interpret_feats.py --ckpt <final> --tags <feature_loop_tags.npz> --device cuda:0 --out <stem>
"""

import argparse
import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sae_lens.saes.sae import TrainingSAE
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.constants import ACTIVATION_SCALER_CFG_FILENAME

from models import get_adapter
from paths import DATA

DOMAINS = ["web", "code", "math", "gsm8k_full", "wiki"]


def collect(ad, sae, scaler, loop, feats, docs_per_domain, topm, ctx, seed, device):
    """For the given feature indices, top-M max-activating contexts at `loop`, plus per-domain firing counts."""
    feats = sorted(int(f) for f in feats)
    fidx = {f: i for i, f in enumerate(feats)}
    heaps = [[] for _ in feats]
    docs: dict[int, tuple] = {}
    uid = 0
    fire = np.zeros((len(DOMAINS), len(feats)))
    n_tok = np.zeros(len(DOMAINS))
    sel = torch.tensor(feats, device=device)
    for di, dom in enumerate(DOMAINS):
        df = pd.read_parquet(DATA / f"domain_eval/{dom}.parquet")
        df = df.sample(min(docs_per_domain, len(df)), random_state=seed)
        for _, row in df.iterrows():
            tok_ids = list(row["input_ids"])
            ids = torch.tensor(tok_ids, device=device).unsqueeze(0)
            with torch.no_grad():
                st = ad.run_states(ids, loop, seed=seed)[loop - 1, 1:].float()
                f = sae.encode(scaler(st))[:, sel]  # (T-1, n_feats)
            fire[di] += (f > 0).sum(0).cpu().numpy()
            n_tok[di] += f.shape[0]
            vals, _ = f.max(0)
            for j in torch.where(vals > 0)[0].tolist():
                v = vals[j].item()
                h = heaps[j]
                if len(h) < topm:
                    heapq.heappush(h, (v, uid))
                    docs[uid] = (dom, tok_ids, int(f[:, j].argmax().item()) + 1)
                    uid += 1
                elif v > h[0][0]:
                    _, old = heapq.heapreplace(h, (v, uid))
                    docs.pop(old, None)
                    docs[uid] = (dom, tok_ids, int(f[:, j].argmax().item()) + 1)
                    uid += 1
        print(f"  {dom}: done", flush=True)

    def ctx_text(tok_ids, pos):
        lo, hi = max(0, pos - ctx), min(len(tok_ids), pos + ctx + 1)
        return f"{ad.tok.decode(tok_ids[lo:pos])}[[{ad.tok.decode(tok_ids[pos:pos+1])}]]{ad.tok.decode(tok_ids[pos+1:hi])}"

    out = {}
    for f in feats:
        j = fidx[f]
        out[str(f)] = {
            "domain_share": dict(zip(DOMAINS, (fire[:, j] / max(fire[:, j].sum(), 1)).round(3).tolist())),
            "top": [{"value": round(v, 3), "domain": docs[u][0], "text": ctx_text(docs[u][1], docs[u][2])}
                    for v, u in sorted(heaps[j], reverse=True) if u in docs][:6],
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tags", required=True, help="feature_loop_tags.npz")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--docs_per_domain", type=int, default=150)
    p.add_argument("--topm", type=int, default=12)
    p.add_argument("--ctx", type=int, default=12)
    p.add_argument("--thr", type=float, default=1e-3)
    p.add_argument("--ratio", type=float, default=5.0)
    p.add_argument("--max_feats", type=int, default=400, help="cap features per class (random subset) for speed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    z = np.load(args.tags)
    F = z["freq_by_loop"]  # (n_loops, d_sae); loops include 1..32
    loops = list(z["loops"])
    e, c = F[loops.index(1)], F[loops.index(32)]
    early = np.where((e > args.thr) & (c < e / args.ratio))[0]
    late = np.where((c > args.thr) & (e < c / args.ratio))[0]
    rng = np.random.default_rng(args.seed)
    early_s = sorted(rng.choice(early, min(args.max_feats, len(early)), replace=False).tolist())
    late_s = sorted(rng.choice(late, min(args.max_feats, len(late)), replace=False).tolist())
    print(f"early {len(early)} (showing {len(early_s)}), late {len(late)} (showing {len(late_s)})", flush=True)

    ckpt = Path(args.ckpt)
    sae = TrainingSAE.load_from_disk(str(ckpt), device=args.device).eval()
    scaler = ActivationScaler()
    scaler.load(ckpt / ACTIVATION_SCALER_CFG_FILENAME)
    ad = get_adapter("huginn", dtype=torch.bfloat16, device=args.device)

    print("interpreting EARLY features at loop 1", flush=True)
    early_out = collect(ad, sae, scaler, 1, early_s, args.docs_per_domain, args.topm, args.ctx, args.seed, args.device)
    print("interpreting LATE features at loop 32", flush=True)
    late_out = collect(ad, sae, scaler, 32, late_s, args.docs_per_domain, args.topm, args.ctx, args.seed, args.device)

    res = {"n_early": int(len(early)), "n_late": int(len(late)), "early": early_out, "late": late_out}
    json.dump(res, open(Path("../results/sae") / f"{args.out}.json", "w"), indent=1, ensure_ascii=False)
    print("saved", Path("../results/sae") / f"{args.out}.json", flush=True)


if __name__ == "__main__":
    main()
