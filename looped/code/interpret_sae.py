"""Find interpretable SAE features: for a (possibly mid-training) checkpoint, collect each latent's maximally
activating token contexts across the per-domain eval corpora, at a chosen loop.

For every document we run Huginn once (run_states, a fixed seed), take the loop state at --loop, encode it with the
SAE (with the checkpoint's activation scaler), and for every latent keep the top-M (activation, doc, position)
triples via a running per-feature min-heap. Output: for a sample of live features, the top contexts decoded to text
with the firing token marked, plus each feature's firing rate and mean activation. Also writes the full top-M table
(feature, value, domain, doc, pos) to npz for later per-domain / per-loop analysis.

CPU-friendly: uses run_states (no attention recompute) and processes one document at a time.

Usage: python interpret_sae.py --ckpt <resume_dir> --loop 8 --device cpu --out <json>
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--loop", type=int, default=8, help="which loop's state to interpret")
    p.add_argument("--device", default="cpu")
    p.add_argument("--docs_per_domain", type=int, default=120)
    p.add_argument("--topm", type=int, default=20, help="top contexts kept per feature")
    p.add_argument("--n_show", type=int, default=40, help="features to print")
    p.add_argument("--ctx", type=int, default=12, help="tokens of context on each side")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_num_threads(32)

    ckpt = Path(args.ckpt)
    sae = TrainingSAE.load_from_disk(str(ckpt), device=args.device).eval()
    scaler = ActivationScaler()
    scaler.load(ckpt / ACTIVATION_SCALER_CFG_FILENAME)
    d_sae = sae.cfg.d_sae
    ad = get_adapter("huginn", dtype=torch.float32 if args.device == "cpu" else torch.bfloat16, device=args.device)

    heaps: list[list] = [[] for _ in range(d_sae)]  # per feature: min-heap of (value, uid)
    fire = np.zeros(d_sae)
    act_sum = np.zeros(d_sae)
    n_tok = 0
    docs = {}  # uid -> (domain, token ids list, pos)
    uid = 0

    for dom in DOMAINS:
        df = pd.read_parquet(DATA / f"domain_eval/{dom}.parquet")
        df = df.sample(min(args.docs_per_domain, len(df)), random_state=args.seed)
        for _, row in df.iterrows():
            ids = torch.tensor(list(row["input_ids"]), device=args.device).unsqueeze(0)
            with torch.no_grad():
                st = ad.run_states(ids, args.loop, seed=args.seed)[args.loop - 1, 1:]  # (T-1, d), drop BOS
                f = sae.encode(scaler(st.float()))  # (T-1, d_sae)
            tok_ids = list(row["input_ids"])
            nz = (f > 0)
            fire += nz.sum(0).cpu().numpy()
            act_sum += f.sum(0).cpu().numpy()
            n_tok += f.shape[0]
            vals, feats = f.max(0)  # strongest token per feature in this doc (cheap top-M feed)
            fired = torch.where(vals > 0)[0]
            for feat in fired.tolist():
                v = vals[feat].item()
                pos = int(f[:, feat].argmax().item()) + 1  # +1: BOS was dropped
                h = heaps[feat]
                if len(h) < args.topm:
                    heapq.heappush(h, (v, uid))
                    docs[uid] = (dom, tok_ids, pos)
                    uid += 1
                elif v > h[0][0]:
                    _, old = heapq.heapreplace(h, (v, uid))
                    docs.pop(old, None)
                    docs[uid] = (dom, tok_ids, pos)
                    uid += 1
        print(f"{dom}: done ({n_tok} tokens so far)", flush=True)

    freq = fire / max(n_tok, 1)
    live = np.where(freq > 1e-4)[0]
    rng = np.random.default_rng(args.seed)
    show = sorted(rng.choice(live, min(args.n_show, len(live)), replace=False).tolist())

    def ctx_text(tok_ids, pos):
        lo, hi = max(0, pos - args.ctx), min(len(tok_ids), pos + args.ctx + 1)
        before = ad.tok.decode(tok_ids[lo:pos])
        tokn = ad.tok.decode(tok_ids[pos:pos + 1])
        after = ad.tok.decode(tok_ids[pos + 1:hi])
        return f"{before}[[{tokn}]]{after}"

    out = {"ckpt": str(ckpt), "loop": args.loop, "n_tokens": n_tok, "n_live_features(freq>1e-4)": int(len(live)),
           "features": {}}
    for feat in show:
        top = sorted(heaps[feat], reverse=True)
        out["features"][str(feat)] = {
            "freq": round(float(freq[feat]), 5),
            "mean_act_when_live": round(float(act_sum[feat] / max(fire[feat], 1)), 3),
            "top_contexts": [{"value": round(v, 3), "domain": docs[u][0], "text": ctx_text(docs[u][1], docs[u][2])}
                             for v, u in top if u in docs][:8],
        }
    json.dump(out, open(args.out, "w"), indent=1, ensure_ascii=False)

    # full top-M table for later analysis -> /data_fs (large), next to the SAE
    rows = [(feat, v, docs[u][0], u, docs[u][2]) for feat in range(d_sae) for v, u in heaps[feat] if u in docs]
    if rows:
        arr = np.array([(f, v, p) for f, v, _, _, p in rows], dtype=np.float32)
        doms = np.array([r[2] for r in rows])
        npz_dir = Path("/data_fs/zhangyuxuan/looped/sae/interp")
        npz_dir.mkdir(parents=True, exist_ok=True)
        np.savez(npz_dir / (Path(args.out).stem + ".npz"), feat_val_pos=arr, domain=doms, freq=freq)

    print(f"\n{len(live)} live features (freq>1e-4) of {d_sae}; showing {len(show)}:\n")
    for feat in show:
        info = out["features"][str(feat)]
        print(f"--- feature {feat}  freq {info['freq']:.4f}  mean_act {info['mean_act_when_live']}")
        for c in info["top_contexts"][:4]:
            print(f"    [{c['value']:6.2f}] ({c['domain']:10s}) {c['text'][:140]}")


if __name__ == "__main__":
    main()
