"""Sanity checks per model: greedy continuation, adapter consistency, CE vs number of loops.

Usage: python sanity.py --model {ouro,huginn}

- Adapter trace has loops * n_core residuals; last-loop readout logits equal the model's own final logits.
- Huginn: recomputed attention reproduces the attention module output (relative error), and the trace is
  reproducible for a fixed seed / differs across seeds (random initial state).
"""

import argparse

import torch
import torch.nn.functional as F

from models import get_adapter

TEXT = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips "
    "to her friends in May. How many clips did Natalia sell altogether in April and May? "
    "Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May."
)


@torch.no_grad()
def greedy(ad, prompt, loops, n=12):
    ids = ad.encode(prompt)
    for _ in range(n):
        tr = ad.run(ids, loops)
        ids = torch.cat([ids, tr.loop_logits(loops - 1)[-1].argmax().view(1, 1)], 1)
    return ad.tok.decode(ids[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ouro", choices=["ouro", "huginn"])
    args = p.parse_args()
    ad = get_adapter(args.model)
    T = ad.train_loops
    print("structure", ad.structure(T))
    print("greedy:", repr(greedy(ad, "The capital of France is", T)))

    ids = ad.encode(TEXT)
    print("bos prepended:", ids[0, 0].item() == ad.tok.bos_token_id, "T =", ids.shape[1])
    kw = {"check": True} if args.model == "huginn" else {}
    tr = ad.run(ids, T, **kw)
    assert tr.resid.shape[0] == T * ad.n_core
    print("resid", tuple(tr.resid.shape), "attn", tuple(tr.attn.shape))
    rows = tr.attn.float().sum(-1)
    print("attn row-sum err", (rows - 1).abs().max().item())
    if args.model == "huginn":
        print("attn recompute rel err (max over layers):", tr.attn_check_rel_err)
        ref = ad.final_logits(ids, T, seed=0)
        print("last-loop readout vs model logits, max abs diff:", (tr.loop_logits(T - 1).float() - ref.float()).abs().max().item())
        tr2 = ad.run(ids, T, seed=0)
        tr3 = ad.run(ids, T, seed=1)
        print("same seed identical:", torch.equal(tr.resid, tr2.resid),
              "| diff seed rel diff:", ((tr.resid - tr3.resid).float().norm() / tr.resid.float().norm()).item())
    else:
        ref = ad.model(ids, use_cache=False).logits[0]
        print("last-loop readout vs model logits, max abs diff:", (tr.loop_logits(T - 1).float() - ref.float()).abs().max().item())

    for loops in sorted({1, 2, 4, 8, 16, 32, T, 2 * T, 4 * T}):
        tr = ad.run(ids, loops)
        ce = F.cross_entropy(tr.loop_logits(loops - 1)[:-1].float(), ids[0, 1:]).item()
        print(f"loops={loops:4d} ({loops / T:5.2f}x train)  CE={ce:.4f}")
        del tr
        torch.cuda.empty_cache()
    print("max mem GB", torch.cuda.max_memory_allocated() / 1e9)


if __name__ == "__main__":
    main()
