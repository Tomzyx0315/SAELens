"""Cache a fixed held-out evaluation set of Huginn loop states (+ loop/seed tags) for the SAE config sweep.

A single fixed eval set makes all configs and all training stages directly comparable. Uses a shuffle seed
disjoint from training, and a few documents; saved to /data_fs. Small (~150k vectors x 5280 x fp16 ~ 1.6 GB).
"""

import argparse

import numpy as np
import torch

from sae_data import LOOPS, HuginnActivationProvider
from paths import BIG


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=64)
    p.add_argument("--context_size", type=int, default=512)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", default=str(BIG / "sae/eval_cache.pt"))
    args = p.parse_args()

    prov = HuginnActivationProvider(train_batch_size=8192, context_size=args.context_size, n_seeds=2,
                                    device=f"cuda:{args.gpu}", producer_gpus=(args.gpu,), buffer_vectors=10**9,
                                    max_rows=args.rows, shuffle_seed=777, split="heldout")
    xs, tags = [], []
    for b in prov:
        xs.append(b.half().cpu())
        tags.append(prov.last_tags.copy())
    prov.close()
    X = torch.cat(xs)
    T = np.concatenate(tags)
    torch.save({"X": X, "tags": torch.from_numpy(T), "loops": list(LOOPS)}, args.out)
    print("cached", tuple(X.shape), "->", args.out)
    print("per-loop counts", {int(l): int((T[:, 0] == l).sum()) for l in LOOPS})


if __name__ == "__main__":
    main()
