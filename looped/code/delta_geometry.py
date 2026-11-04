"""Is adjacent-layer CKA high just because of the residual stream (x_l = x_{l-1} + update)?

On the geometry token pools (geometry.py), compare per pair of depths:

- CKA / centered cosine of the residual x_d
- CKA / centered cosine of the per-layer update u_d = x_d - x_{d-1} (within a loop; layer 0 of each loop excluded)

and report the relative update size ||u_d|| / ||x_d||. If adjacent-layer residual similarity is mostly
inherited through the residual stream, update-CKA between adjacent layers should be much lower than
residual-CKA, while the same layer across loops (same weights) may stay high.

Usage: python delta_geometry.py --model {ouro,huginn} --loops 8
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from paths import raw_dir, results_dir

N_CORE = {"ouro": 24, "huginn": 4}


def cka_matrix(Xc: torch.Tensor) -> torch.Tensor:
    """Xc: (D_, N, d) centered. Linear CKA between all depth pairs via flattened token Gram matrices."""
    G = torch.stack([(x @ x.T).flatten() for x in Xc])
    G = F.normalize(G, dim=-1)
    return G @ G.T


def cos_matrix(Xc: torch.Tensor) -> torch.Tensor:
    Xn = F.normalize(Xc, dim=-1)
    return torch.einsum("and,bnd->ab", Xn, Xn) / Xc.shape[1]


def summarize(m: np.ndarray, valid: np.ndarray, nl: int, loops: int) -> dict:
    """valid: depths that have an update (layer >= 1). Summaries restricted to loops >= 3 (t >= 2, 0-indexed)."""
    same, adj, far = [], [], []
    for t in range(2, loops):
        for l in range(1, nl):
            d = t * nl + l
            if t + 1 < loops:
                same.append(m[d, d + nl])
            if l + 1 < nl:
                adj.append(m[d, d + 1])
            if nl > 4:
                l2 = 1 + (l - 1 + (nl - 1) // 2) % (nl - 1)
                far.append(m[d, t * nl + l2])
    hits, tot = 0, 0
    for d in np.where(valid)[0]:
        if d < 2 * nl:
            continue
        r = m[d].copy()
        r[~valid] = -np.inf
        r[d] = -np.inf
        hits += (r.argmax() % nl) == (d % nl)
        tot += 1
    out = {"same_layer_next_loop": float(np.mean(same)), "adjacent_layer": float(np.mean(adj)),
           "nearest_is_same_layer_frac": hits / max(tot, 1)}
    if far:
        out["half_block_apart"] = float(np.mean(far))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ouro", choices=["ouro", "huginn"])
    p.add_argument("--loops", type=int, default=8)
    p.add_argument("--threads", type=int, default=64)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    nl, loops = N_CORE[args.model], args.loops

    X = torch.load(raw_dir(args.model) / f"geometry_pool_gsm8k_L{loops}.pt").float()  # (D_, N, d)
    D_ = X.shape[0]
    assert D_ == nl * loops
    layer = np.arange(D_) % nl
    valid = layer >= 1
    U = torch.zeros_like(X)
    U[1:] = X[1:] - X[:-1]
    U[~torch.from_numpy(valid)] = 0  # layer 0: previous depth is across the end-of-loop norm / adapter
    rel_update = (U.norm(dim=-1).mean(-1) / X.norm(dim=-1).mean(-1)).numpy()

    Xc = X - X.mean(1, keepdim=True)
    Uc = U - U.mean(1, keepdim=True)
    res = {}
    for name, Z in (("resid", Xc), ("update", Uc)):
        for kind, fn in (("cka", cka_matrix), ("cen_cos", cos_matrix)):
            m = fn(Z).numpy()
            res[f"{name}_{kind}"] = m
            print(f"{args.model} L{loops} {name:6s} {kind:7s}", {k: round(v, 3) for k, v in summarize(m, valid, nl, loops).items()},
                  flush=True)
    rel = rel_update.reshape(loops, nl)[:, 1:]
    print("relative update ||u||/||x|| by loop (mean over layers>=1):", rel.mean(1).round(3).tolist())
    np.savez(results_dir(args.model) / f"delta_geometry_gsm8k_L{loops}.npz", rel_update=rel_update, valid=valid, **res)


if __name__ == "__main__":
    main()
