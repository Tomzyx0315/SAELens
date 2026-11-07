"""Quantitative, LLM-free descriptors of SAE features, to compare early / late-emerging / persistent features (A4).

For every latent, over all tokens where it fires (at the loop where its class is defined: early at loop 1,
late and persistent at loop 32), we compute:

- freq: firing rate
- token_entropy: normalised entropy (bits / log2(#distinct)) of the distribution of the *current token* it fires on
  (low = token-specific, e.g. fires on one word; high = context-driven)
- top1_token_share: share of firings on its single most common current token
- rel_pos_mean: mean relative position (pos / doc length) of firings (0.5 = position-neutral)
- domain_entropy: normalised entropy over the 5 domains (low = domain-specific)
- char_class shares of the current token: alpha word / digit / punctuation / whitespace / code-symbol
- answer_share (math / gsm8k): fraction of firings on answer (solution) tokens vs question tokens

Class definitions from feature_loop_tags.npz (same as SAE.md). Reports per-class medians and Mann-Whitney U
p-values for early vs late. Writes results/sae/feature_descriptors.json.

Usage: python feature_descriptors.py --ckpt <final> --tags <feature_loop_tags.npz> --device cuda:3 --out feature_descriptors
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu

from sae_lens.constants import ACTIVATION_SCALER_CFG_FILENAME
from sae_lens.saes.sae import TrainingSAE
from sae_lens.training.activation_scaler import ActivationScaler

from models import get_adapter
from paths import DATA

DOMAINS = ["web", "code", "math", "gsm8k_full", "wiki"]
MAX_FIRINGS = 400  # per feature, reservoir-sampled, for the token-level descriptors


def char_class(s):
    t = s.strip()
    if not t:
        return "space"
    if t.isdigit():
        return "digit"
    if t.isalpha():
        return "alpha"
    if all(not c.isalnum() for c in t):
        return "punct"
    return "mixed"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tags", required=True)
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--docs_per_domain", type=int, default=200)
    p.add_argument("--thr", type=float, default=1e-3)
    p.add_argument("--ratio", type=float, default=5.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    z = np.load(args.tags)
    F = z["freq_by_loop"]
    loops = list(z["loops"])
    e, c = F[loops.index(1)], F[loops.index(32)]
    cls = {"early": np.where((e > args.thr) & (c < e / args.ratio))[0],
           "late": np.where((c > args.thr) & (e < c / args.ratio))[0]}
    cls["persistent"] = np.where((e > args.thr) & (c > args.thr) & ~np.isin(np.arange(len(e)), np.r_[cls["early"], cls["late"]]))[0]
    at_loop = {"early": 1, "late": 32, "persistent": 32}

    ckpt = Path(args.ckpt)
    sae = TrainingSAE.load_from_disk(str(ckpt), device=args.device).eval()
    scaler = ActivationScaler()
    scaler.load(ckpt / ACTIVATION_SCALER_CFG_FILENAME)
    ad = get_adapter("huginn", dtype=torch.bfloat16, device=args.device)
    vocab_str = {}

    # per (loop, feature): reservoir of (token_id, rel_pos, domain_idx, is_answer) + counts
    rng = np.random.default_rng(0)
    feats_by_loop = {l: np.unique(np.concatenate([cls[k] for k in cls if at_loop[k] == l])) for l in (1, 32)}
    res_store = {l: {int(f): [] for f in feats_by_loop[l]} for l in (1, 32)}
    seen = {l: {int(f): 0 for f in feats_by_loop[l]} for l in (1, 32)}
    dom_cnt = {l: {int(f): np.zeros(len(DOMAINS)) for f in feats_by_loop[l]} for l in (1, 32)}
    n_tok = 0

    for di, dom in enumerate(DOMAINS):
        df = pd.read_parquet(DATA / f"domain_eval/{dom}.parquet")
        df = df.sample(min(args.docs_per_domain, len(df)), random_state=2)
        for _, row in df.iterrows():
            tok_ids = list(row["input_ids"])
            ids = torch.tensor(tok_ids, device=args.device).unsqueeze(0)
            ans = int(row["answer_start"])
            T = len(tok_ids) - 1
            n_tok += T
            with torch.no_grad():
                st = ad.run_states(ids, 32, seed=0)[:, 1:]
                for l in (1, 32):
                    fl = feats_by_loop[l]
                    a = sae.encode(scaler(st[l - 1].float()))[:, torch.tensor(fl, device=args.device)] > 0
                    tpos, fidx = torch.where(a)
                    for tp, fi in zip(tpos.tolist(), fidx.tolist()):
                        f = int(fl[fi])
                        pos = tp + 1  # absolute position in tok_ids
                        dom_cnt[l][f][di] += 1
                        rec = (tok_ids[pos], pos / len(tok_ids), di, int(ans >= 0 and pos >= ans))
                        seen[l][f] += 1
                        if len(res_store[l][f]) < MAX_FIRINGS:
                            res_store[l][f].append(rec)
                        else:
                            k = rng.integers(0, seen[l][f])
                            if k < MAX_FIRINGS:
                                res_store[l][f][k] = rec
        print(f"{dom}: done", flush=True)

    def tok_str(t):
        if t not in vocab_str:
            vocab_str[t] = ad.tok.decode([t])
        return vocab_str[t]

    def describe(l, f):
        recs = res_store[l][f]
        if len(recs) < 20:
            return None
        toks = Counter(r[0] for r in recs)
        probs = np.array(list(toks.values()), float) / len(recs)
        ent = -(probs * np.log2(probs)).sum() / max(np.log2(len(recs)), 1e-9)
        cc = Counter(char_class(tok_str(r[0])) for r in recs)
        dc = dom_cnt[l][f]
        dp = dc / dc.sum()
        dent = -(dp[dp > 0] * np.log2(dp[dp > 0])).sum() / np.log2(len(DOMAINS))
        mrec = [r for r in recs if DOMAINS[r[2]] in ("math", "gsm8k_full")]
        return {
            "n_firings": int(seen[l][f]), "freq": seen[l][f] / n_tok,
            "token_entropy": float(ent), "top1_token_share": float(probs.max()),
            "rel_pos_mean": float(np.mean([r[1] for r in recs])),
            "domain_entropy": float(dent),
            **{f"cc_{k}": cc.get(k, 0) / len(recs) for k in ("alpha", "digit", "punct", "space", "mixed")},
            "answer_share": float(np.mean([r[3] for r in mrec])) if len(mrec) >= 10 else None,
            "top_tokens": [tok_str(t) for t, _ in toks.most_common(5)],
        }

    out = {"n_tokens": int(n_tok), "classes": {}, "tests_early_vs_late": {}}
    desc = {}
    for k in cls:
        l = at_loop[k]
        desc[k] = [d for d in (describe(l, int(f)) for f in cls[k]) if d is not None]
        keys = [x for x in desc[k][0] if x not in ("top_tokens", "n_firings")]
        summ = {}
        for x in keys:
            vals = [d[x] for d in desc[k] if d[x] is not None]
            if vals:
                summ[x] = {"median": round(float(np.median(vals)), 4), "mean": round(float(np.mean(vals)), 4)}
        out["classes"][k] = {"n_features": int(len(cls[k])), "n_described": len(desc[k]), "summary": summ,
                             "examples": [{"top_tokens": d["top_tokens"], "token_entropy": round(d["token_entropy"], 3),
                                           "freq": round(d["freq"], 5)} for d in desc[k][:15]]}
    for x in out["classes"]["early"]["summary"]:
        a = [d[x] for d in desc["early"] if d.get(x) is not None]
        b = [d[x] for d in desc["late"] if d.get(x) is not None]
        if len(a) > 10 and len(b) > 10:
            out["tests_early_vs_late"][x] = {"p_mannwhitney": float(mannwhitneyu(a, b).pvalue),
                                             "median_early": round(float(np.median(a)), 4),
                                             "median_late": round(float(np.median(b)), 4)}
    path = Path(__file__).resolve().parent.parent / "results/sae" / f"{args.out}.json"
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"tokens {n_tok}; described early {len(desc['early'])} late {len(desc['late'])} persistent {len(desc['persistent'])}")
    print(f"{'descriptor':18s} {'early':>8} {'late':>8} {'persist':>8}  p(early vs late)")
    for x, t in out["tests_early_vs_late"].items():
        pm = out["classes"]["persistent"]["summary"].get(x, {}).get("median", float("nan"))
        print(f"{x:18s} {t['median_early']:8.4f} {t['median_late']:8.4f} {pm:8.4f}  {t['p_mannwhitney']:.1e}")
    print("saved", path)


if __name__ == "__main__":
    main()
