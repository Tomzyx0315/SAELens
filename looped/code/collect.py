"""Collect per-(loop, layer) residual / attention statistics from a looped model (or feedforward baseline).

One forward pass per example (batch size 1, no padding). Depth d = loop * n_core + layer_in_block, covering the
recurrent block only (prelude / coda layers are not included). Per example everything is reduced to
depth-indexed metrics and saved as RAW_ROOT/<model>/<run>/ex*.npz; structure info goes to args.json.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from metrics import attn_metrics, load_texts, residual_stats, tok_cos
from models import get_adapter
from paths import raw_dir


@torch.no_grad()
def process(trace, ids, loops, n_core, sink_pos, pairwise, save_last_token):
    R, A = trace.resid, trace.attn
    D_ = loops * n_core
    T = R.shape[1]
    res: dict[str, np.ndarray] = {}

    # ---- per-depth attention + residual metrics (chunked to bound memory) ----
    chunks: dict[str, list] = {}
    for s in range(0, D_, 96):
        m = attn_metrics(A[s : s + 96].float(), sink_pos)
        m.update(residual_stats(R[s : s + 96]))
        if trace.resid_pre_norm is not None:
            pre = residual_stats(trace.resid_pre_norm[s : s + 96])
            m.update({f"pre_{k}": v for k, v in pre.items()})
        for k, v in m.items():
            chunks.setdefault(k, []).append(v)
    for k, v in chunks.items():
        res[k] = torch.cat(v).cpu().numpy()

    # ---- per-layer recurrence dynamics: successive & distance to last loop ("approx fixed point") ----
    Rl = R.view(loops, n_core, T, -1)
    Al = A.view(loops, n_core, *A.shape[1:])
    dyn = {k: torch.zeros(loops, n_core) for k in ("fp_norm", "fp_cos", "fp_attn", "succ_norm", "succ_cos", "succ_attn")}
    for l in range(n_core):
        x = Rl[:, l].float()  # (loops, T, D)
        a = Al[:, l].float()  # (loops, H, T, T)
        dyn["fp_norm"][:, l] = (x - x[-1]).norm(dim=-1).mean(-1).cpu()
        dyn["fp_cos"][:, l] = tok_cos(x, x[-1:]).mean(-1).cpu()
        dyn["fp_attn"][:, l] = (a - a[-1]).flatten(2).norm(dim=-1).mean(-1).cpu()
        if loops > 1:
            dyn["succ_norm"][1:, l] = (x[1:] - x[:-1]).norm(dim=-1).mean(-1).cpu()
            dyn["succ_cos"][1:, l] = tok_cos(x[1:], x[:-1]).mean(-1).cpu()
            dyn["succ_attn"][1:, l] = (a[1:] - a[:-1]).flatten(2).norm(dim=-1).mean(-1).cpu()
    res.update({k: v.numpy() for k, v in dyn.items()})

    # ---- per-loop readout CE with the model's own head ----
    if T > 1:
        res["loop_ce"] = np.array(
            [F.cross_entropy(trace.loop_logits(t)[:-1].float(), ids[0, 1:]).item() for t in range(loops)]
        )
    res["loop_state_norm"] = trace.loop_state.float().norm(dim=-1).mean(-1).cpu().numpy()

    # ---- pairwise depth x depth heatmaps (only for small loop counts) ----
    if pairwise:
        Xn = F.normalize(R.float(), dim=-1)  # (D_, T, D)
        res["pair_cos"] = (torch.einsum("atd,btd->ab", Xn, Xn) / T).cpu().numpy()
        Af = A.float().flatten(2).transpose(0, 1)  # (H, D_, T*T)
        res["pair_attn"] = torch.cdist(Af, Af).mean(0).cpu().numpy()

    if save_last_token:
        res["last_tok_resid"] = R[:, -1].float().cpu().numpy()
        res["last_tok_state"] = trace.loop_state[:, -1].float().cpu().numpy()
    res["T"] = np.array(T)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ouro", choices=["ouro", "huginn", "llama"])
    p.add_argument("--data", default="gsm8k")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--loops", type=int, default=4)
    p.add_argument("--run", default=None, help="run name; default <data>_L<loops>")
    p.add_argument("--pairwise", action="store_true")
    p.add_argument("--save_last_token", action="store_true")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--seed_offset", type=int, default=0, help="Huginn only: init-state seed = example idx + offset")
    p.add_argument("--overwrite", action="store_true", help="recompute existing ex*.npz (default: resume)")
    args = p.parse_args()

    run = args.run or f"{args.data}_L{args.loops}"
    out = raw_dir(args.model) / run
    os.makedirs(out, exist_ok=True)
    ad = get_adapter(args.model, dtype=getattr(torch, args.dtype))
    st = ad.structure(args.loops)
    texts = load_texts(args.data, args.n)
    with open(out / "args.json", "w") as f:
        json.dump({**vars(args), "run": run, "structure": st}, f)
    for i in range(args.shard, len(texts), args.nshards):
        path = out / f"ex{i:04d}.npz"
        if path.exists() and not args.overwrite:
            continue
        ids = ad.encode(texts[i])[:, : args.max_tokens]
        kw = {"seed": i + args.seed_offset} if args.model == "huginn" else {}
        trace = ad.run(ids, args.loops, **kw)
        res = process(trace, ids, args.loops, st["n_core"], st["sink_pos"], args.pairwise, args.save_last_token)
        np.savez(path, **res)
        del trace
        torch.cuda.empty_cache()
        print(f"[{args.shard}] {i} T={ids.shape[1]} done", flush=True)


if __name__ == "__main__":
    main()
