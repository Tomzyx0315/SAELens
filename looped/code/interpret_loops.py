"""Interpret SAE features at several loops from a single forward pass per document (CPU-friendly).

For each eval document we run Huginn once at max(LOOPS) loops (fixed seed), encode the loop state at every target
loop with the SAE (+ the checkpoint's scaler), and for every (loop, feature) keep the top-M activating token contexts
via running min-heaps. Also accumulate per-(loop, feature, domain) firing counts. This supports comparing the
semantics of "early" (active at loop 1, not later) vs "late-emerging" (active at the converged loop, not early)
features, and per-domain loop analysis.

Output: `<out>.json` with, per loop, the top contexts of a sample of features; and `/data_fs/.../interp/<out>.npz`
with the full per-(loop, feature) firing frequencies by domain and the top-context table.

Usage: python interpret_loops.py --ckpt <resume_dir> --loops 1,8,32 --device cpu --out <stem>
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
NPZ_DIR = Path("/data_fs/zhangyuxuan/looped/sae/interp")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--loops", default="1,8,32")
    p.add_argument("--device", default="cpu")
    p.add_argument("--docs_per_domain", type=int, default=100)
    p.add_argument("--topm", type=int, default=16)
    p.add_argument("--ctx", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="stem; json in results/sae/, npz in /data_fs/.../interp/")
    args = p.parse_args()
    torch.set_num_threads(48)
    loops = [int(x) for x in args.loops.split(",")]
    max_loop = max(loops)

    ckpt = Path(args.ckpt)
    sae = TrainingSAE.load_from_disk(str(ckpt), device=args.device).eval()
    scaler = ActivationScaler()
    scaler.load(ckpt / ACTIVATION_SCALER_CFG_FILENAME)
    d_sae = sae.cfg.d_sae
    ad = get_adapter("huginn", dtype=torch.float32 if args.device == "cpu" else torch.bfloat16, device=args.device)

    heaps = {l: [[] for _ in range(d_sae)] for l in loops}  # (loop)[feat] -> min-heap (value, uid)
    docs: dict[int, tuple] = {}
    uid = 0
    fire = {l: np.zeros((len(DOMAINS), d_sae)) for l in loops}
    n_tok = {l: np.zeros(len(DOMAINS)) for l in loops}

    for di, dom in enumerate(DOMAINS):
        df = pd.read_parquet(DATA / f"domain_eval/{dom}.parquet")
        df = df.sample(min(args.docs_per_domain, len(df)), random_state=args.seed)
        for _, row in df.iterrows():
            tok_ids = list(row["input_ids"])
            ids = torch.tensor(tok_ids, device=args.device).unsqueeze(0)
            with torch.no_grad():
                st = ad.run_states(ids, max_loop, seed=args.seed)  # (max_loop, T, d)
                for l in loops:
                    f = sae.encode(scaler(st[l - 1, 1:].float()))  # (T-1, d_sae), drop BOS
                    nz = (f > 0)
                    fire[l][di] += nz.sum(0).cpu().numpy()
                    n_tok[l][di] += f.shape[0]
                    vals, _ = f.max(0)
                    for feat in torch.where(vals > 0)[0].tolist():
                        v = vals[feat].item()
                        h = heaps[l][feat]
                        if len(h) < args.topm:
                            heapq.heappush(h, (v, uid))
                            docs[uid] = (dom, tok_ids, int(f[:, feat].argmax().item()) + 1)
                            uid += 1
                        elif v > h[0][0]:
                            _, old = heapq.heapreplace(h, (v, uid))
                            docs.pop(old, None)
                            docs[uid] = (dom, tok_ids, int(f[:, feat].argmax().item()) + 1)
                            uid += 1
        print(f"{dom}: done", flush=True)

    def ctx_text(tok_ids, pos):
        lo, hi = max(0, pos - args.ctx), min(len(tok_ids), pos + args.ctx + 1)
        return f"{ad.tok.decode(tok_ids[lo:pos])}[[{ad.tok.decode(tok_ids[pos:pos+1])}]]{ad.tok.decode(tok_ids[pos+1:hi])}"

    freq = {l: fire[l].sum(0) / max(n_tok[l].sum(), 1) for l in loops}  # overall per-loop firing freq
    out = {"ckpt": str(ckpt), "loops": loops, "domains": DOMAINS, "features": {}}
    rng = np.random.default_rng(args.seed)
    live_any = np.where(np.stack([freq[l] for l in loops]).max(0) > 1e-4)[0]
    show = sorted(rng.choice(live_any, min(40, len(live_any)), replace=False).tolist())
    for feat in show:
        out["features"][str(feat)] = {
            "freq_by_loop": {str(l): round(float(freq[l][feat]), 5) for l in loops},
            "top_by_loop": {str(l): [{"value": round(v, 3), "domain": docs[u][0],
                                      "text": ctx_text(docs[u][1], docs[u][2])}
                                     for v, u in sorted(heaps[l][feat], reverse=True) if u in docs][:5]
                            for l in loops},
        }
    json.dump(out, open(Path("../results/sae") / f"{args.out}.json", "w"), indent=1, ensure_ascii=False)

    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    save = {"loops": np.array(loops), "domains": np.array(DOMAINS)}
    for l in loops:
        save[f"freq_dom_{l}"] = fire[l] / np.maximum(n_tok[l][:, None], 1)  # (n_dom, d_sae)
        rows = [(feat, v, docs[u][0], docs[u][2]) for feat in range(d_sae) for v, u in heaps[l][feat] if u in docs]
        save[f"top_{l}_fvp"] = np.array([(f, v, p) for f, v, _, p in rows], dtype=np.float32)
        save[f"top_{l}_dom"] = np.array([r[2] for r in rows])
    # keep token ids of the contexts referenced, so offline analysis can re-decode arbitrary features
    np.savez(NPZ_DIR / f"{args.out}.npz", **save)
    print("saved", Path("../results/sae") / f"{args.out}.json", flush=True)


if __name__ == "__main__":
    main()
