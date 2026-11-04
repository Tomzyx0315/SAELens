"""Residual-stream geometry across (loop, layer) depths, controlling for the shared mean direction.

For 256 GSM8k questions, pool residuals (output of each recurrent-block layer) from up to 16 random
positions per example, excluding position 0 (attention-sink token with massive norm) and, if the model
prepends BOS, also position 1 is kept (only position 0 is excluded). On this pool of N tokens we compute
for every pair of depths (i, j):

- raw cosine:      mean_t cos(x_i[t], x_j[t])
- centered cosine: same after subtracting each depth's pool mean mu_i
- linear CKA:      <Kc_i, Kc_j>_F / (||Kc_i|| ||Kc_j||), Kc = centered token Gram matrix

and per depth the fraction of energy in the mean direction, ||mu||^2 / mean_t ||x[t]||^2.
Attention-sink diagnostics are recorded by collect.py (top_key_is_pos0, sink_score, massive_ratio).

Outputs: results/<model>/geometry_<data>_L<loops>.npz, and the token pool (fp16) on /data_fs for reuse.
"""

import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F

from metrics import load_texts
from models import get_adapter
from paths import raw_dir, results_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ouro", choices=["ouro", "huginn"])
    p.add_argument("--data", default="gsm8k")
    p.add_argument("--loops", type=int, default=8)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--per_example", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=32)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    ad = get_adapter(args.model, dtype=dtype, device=args.device)
    rng = random.Random(0)
    pool = []
    texts = load_texts(args.data, args.n)
    for ei, text in enumerate(texts):
        ids = ad.encode(text)
        kw = {"seed": ei} if args.model == "huginn" else {}
        R = ad.run(ids, args.loops, **kw).resid.float()  # (D_, T, d)
        T = R.shape[1]
        pos = rng.sample(range(1, T), min(args.per_example, T - 1))
        pool.append(R[:, pos].cpu().half())
        if ei % 32 == 0:
            print(f"{ei} T={T}", flush=True)

    X = torch.cat(pool, 1).float()  # (D_, N, d)
    D_, N = X.shape[:2]
    rd = raw_dir(args.model)
    rd.mkdir(parents=True, exist_ok=True)
    torch.save(X.half(), rd / f"geometry_pool_{args.data}_L{args.loops}.pt")
    print("pool", tuple(X.shape), flush=True)

    mu = X.mean(1, keepdim=True)
    mean_energy_frac = (mu.squeeze(1).pow(2).sum(-1) / X.pow(2).sum(-1).mean(-1)).numpy()
    Xn = F.normalize(X, dim=-1)
    raw_cos = torch.einsum("and,bnd->ab", Xn, Xn) / N
    Xc = X - mu
    Xcn = F.normalize(Xc, dim=-1)
    cen_cos = torch.einsum("and,bnd->ab", Xcn, Xcn) / N
    G = torch.empty(D_, N * N)
    for i in range(D_):
        G[i] = (Xc[i] @ Xc[i].T).flatten()
    G = F.normalize(G, dim=-1)
    cka = G @ G.T

    out = results_dir(args.model) / f"geometry_{args.data}_L{args.loops}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, raw_cos=raw_cos.numpy(), cen_cos=cen_cos.numpy(), cka=cka.numpy(),
             mean_energy_frac=mean_energy_frac, N=np.array(N))
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
