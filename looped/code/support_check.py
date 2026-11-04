"""Feasibility check before training SAEs: across loops, does the *set* of active units change, or do the same
units just get stronger?

No SAE is trained here. We use the MLP hidden activation h = act(gate) * up of each recurrent-block layer
(input of the down projection; Huginn 17920-d, Ouro 5632-d) as a high-dimensional, model-privileged basis.
Per token (position 0 excluded) we take the top-k units by |h| and compare these supports across loops with
Jaccard similarity, relative to the reference loop = train depth (Ouro 4, Huginn 32).

Calibration baselines at the reference loop, same layer:

- adjacent tokens: token i vs token i+1 (a genuinely different but related representation)
- distant tokens: token i vs token i + T/2 (cyclic)
- Huginn only: same token, every loop, a second random initial state (how much of the change is init noise)

Also accumulates per-unit top-k firing frequency per (loop, layer) to count units that emerge / vanish over loops,
and the fraction of |h| mass captured by the top-k (whether the basis is sparse to begin with).

Usage: python support_check.py --model huginn --loops 48   |   python support_check.py --model ouro --loops 8
"""

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from metrics import load_texts
from models import get_adapter
from paths import results_dir

KS = (32, 128)


def mlp_modules(ad):
    if ad.name == "huginn":
        return [b.mlp.proj for b in ad.model.transformer.core_block]
    return [layer.mlp.down_proj for layer in ad.model.model.layers]


@torch.no_grad()
def mlp_acts(ad, ids, loops, seed=0):
    """(loops, n_core, T, d_mlp) MLP hidden activations of the recurrent block, loop-major."""
    mods = mlp_modules(ad)
    buf = []
    # offload to CPU immediately: keeps peak GPU memory to the model + one forward (shared GPUs)
    hooks = [m.register_forward_pre_hook(lambda mod, args: buf.append(args[0][0].to("cpu", non_blocking=True)))
             for m in mods]
    try:
        if ad.name == "huginn":
            torch.manual_seed(seed)
            ad.model(ids, num_steps=loops, use_cache=False)
        else:
            ad.model.config.total_ut_steps = loops
            ad.model.model.total_ut_steps = loops
            ad.model(ids, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    nl = len(mods)
    assert len(buf) == loops * nl, len(buf)
    H = torch.stack(buf)
    return H.view(loops, nl, *H.shape[1:])


def topk_mask(A: torch.Tensor, k: int) -> torch.Tensor:
    """A: |h| (..., T, d) float -> bool mask of the top-k units per token."""
    idx = A.topk(k, dim=-1).indices
    m = torch.zeros(A.shape, dtype=torch.bool, device=A.device)
    m.scatter_(-1, idx, True)
    return m


def jacc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    inter = (a & b).sum(-1).float()
    return inter / ((a | b).sum(-1).float())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="huginn", choices=["ouro", "huginn"])
    p.add_argument("--loops", type=int, default=48)
    p.add_argument("--n", type=int, default=256)
    args = p.parse_args()

    ad = get_adapter(args.model)
    ref = ad.train_loops - 1
    L = args.loops
    assert ref < L
    nl = len(mlp_modules(ad))
    acc: dict[str, np.ndarray] = {}
    freq = {k: None for k in KS}
    n_tok = 0

    def add(key, v):
        v = v.cpu().numpy() if torch.is_tensor(v) else v
        acc[key] = acc.get(key, 0) + v

    def layer_stats(A, A2):
        """A, A2: (L, T', d) |h| for one layer. Returns dict of (L,) / (L, d) tensors."""
        T = A.shape[1]
        out = {}
        An = A / A.norm(dim=-1, keepdim=True)
        out["cos_ref"] = (An * An[ref : ref + 1]).sum(-1).mean(-1)
        del An
        for k in KS:
            M = topk_mask(A, k)
            out[f"jac_ref_k{k}"] = jacc(M, M[ref : ref + 1]).mean(-1)
            out[f"jac_succ_k{k}"] = torch.cat([torch.zeros(1, device=M.device), jacc(M[1:], M[:-1]).mean(-1)])
            out[f"jac_adjtok_k{k}"] = jacc(M[:, :-1], M[:, 1:]).mean(-1)
            out[f"jac_fartok_k{k}"] = jacc(M, M.roll(T // 2, dims=1)).mean(-1)
            out[f"mass_k{k}"] = ((A * M).sum(-1) / A.sum(-1)).mean(-1)
            out[f"freq_k{k}"] = M.sum(1).float()  # (L, d)
            if A2 is not None:
                out[f"jac_seed_k{k}"] = jacc(M, topk_mask(A2, k)).mean(-1)
            del M
        return out

    def example_stats(ei, ids):
        H = mlp_acts(ad, ids, L, seed=ei)[:, :, 1:]  # (L, nl, T', d) bf16 on CPU, drop position 0
        H2 = mlp_acts(ad, ids, L, seed=ei + 10_000)[:, :, 1:] if args.model == "huginn" else None
        dev = ad.device
        per_layer = [layer_stats(H[:, l].to(dev).float().abs(), None if H2 is None else H2[:, l].to(dev).float().abs())
                     for l in range(nl)]
        return H.shape[2], per_layer

    texts = load_texts("gsm8k", args.n)
    for ei, text in enumerate(texts):
        ids = ad.encode(text)
        for attempt in range(60):  # GPUs are shared; other jobs spike memory transiently
            try:
                T, per_layer = example_stats(ei, ids)
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"OOM on example {ei}, retry {attempt}", flush=True)
                time.sleep(30)
        else:
            raise RuntimeError(f"example {ei}: persistent OOM")
        for key in per_layer[0]:
            v = torch.stack([s[key] for s in per_layer], 1)  # (L, nl) or (L, nl, d)
            if key.startswith("freq"):
                k = int(key.split("_k")[1])
                freq[k] = v if freq[k] is None else freq[k] + v
            else:
                add(key, v)
        n_tok += T
        del per_layer
        torch.cuda.empty_cache()
        if ei % 32 == 0:
            print(f"{ei} T={T}", flush=True)

    n = len(texts)
    res = {k: v / n for k, v in acc.items()}
    for k in KS:
        res[f"freq_k{k}"] = (freq[k] / n_tok).cpu().numpy()  # (L, nl, d) fraction of tokens with unit in top-k

    # unit-usage turnover relative to the reference loop
    summ = {"model": args.model, "loops": L, "ref_loop": ref + 1, "n_layers": nl, "n_examples": n, "n_tokens": n_tok}
    for k in KS:
        F = res[f"freq_k{k}"]
        P = F / F.sum(-1, keepdims=True)
        res[f"usage_overlap_k{k}"] = np.minimum(P, P[ref : ref + 1]).sum(-1)  # (L, nl), 1 = identical usage
        thr = 0.01
        em, va = np.zeros((L, nl)), np.zeros((L, nl))
        for t in range(L):
            em[t] = ((F[ref] >= thr) & (F[t] < F[ref] / 5)).sum(-1)  # used at ref, (almost) unused at t
            va[t] = ((F[t] >= thr) & (F[ref] < F[t] / 5)).sum(-1)  # used at t, (almost) unused at ref
        res[f"n_emerging_k{k}"], res[f"n_vanishing_k{k}"] = em, va
        res[f"n_used_k{k}"] = (F >= thr).sum(-1)

    def at(key, loops_sel):
        v = res[key]
        return {str(t): [round(float(x), 3) for x in v[t - 1]] for t in loops_sel if t <= L}

    sel = sorted({1, 2, 3, 4, 8, 16, 24, 32, 48, ad.train_loops, L} & set(range(1, L + 1)))
    for k in KS:
        summ[f"k{k}"] = {
            "jaccard_to_ref_per_layer": at(f"jac_ref_k{k}", sel),
            "jaccard_successive_per_layer": at(f"jac_succ_k{k}", sel),
            "baseline_adjacent_tokens_at_ref": [round(float(x), 3) for x in res[f"jac_adjtok_k{k}"][ref]],
            "baseline_distant_tokens_at_ref": [round(float(x), 3) for x in res[f"jac_fartok_k{k}"][ref]],
            "topk_mass_fraction_at_ref": [round(float(x), 3) for x in res[f"mass_k{k}"][ref]],
            "usage_overlap_with_ref": at(f"usage_overlap_k{k}", sel),
            "n_units_used_at_ref(freq>=1%)": res[f"n_used_k{k}"][ref].tolist(),
            "n_emerging(used at ref, <1/5 as often at t)": at(f"n_emerging_k{k}", sel),
            "n_vanishing(used at t, <1/5 as often at ref)": at(f"n_vanishing_k{k}", sel),
        }
        if f"jac_seed_k{k}" in res:
            summ[f"k{k}"]["jaccard_between_init_seeds_per_layer"] = at(f"jac_seed_k{k}", sel)
    summ["cos_abs_h_to_ref_per_layer"] = at("cos_ref", sel)

    out = results_dir(args.model)
    np.savez(out / f"support_check_gsm8k_L{L}.npz", **{k: v for k, v in res.items() if not k.startswith("freq")})
    np.savez_compressed(out / f"support_check_freq_gsm8k_L{L}.npz", **{k: v for k, v in res.items() if k.startswith("freq")})
    json.dump(summ, open(out / f"support_check_gsm8k_L{L}.json", "w"), indent=1)

    # ---- figure ----
    k = 128
    x = np.arange(1, L + 1)
    cm = plt.get_cmap("winter")
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.3))
    for l in range(nl):
        c = cm(l / max(nl - 1, 1))
        axes[0].plot(x, res[f"jac_ref_k{k}"][:, l], color=c, lw=1)
        axes[1].plot(x[1:], res[f"jac_succ_k{k}"][1:, l], color=c, lw=1)
        axes[2].plot(x, res[f"usage_overlap_k{k}"][:, l], color=c, lw=1)
        axes[3].plot(x, res[f"n_emerging_k{k}"][:, l], color=c, lw=1)
        axes[3].plot(x, res[f"n_vanishing_k{k}"][:, l], color=c, lw=1, ls="--")
    for ax in axes[:2]:
        ax.axhline(res[f"jac_adjtok_k{k}"][ref].mean(), color="r", ls=":", label="adjacent tokens @ref")
        ax.axhline(res[f"jac_fartok_k{k}"][ref].mean(), color="m", ls=":", label="distant tokens @ref")
    if f"jac_seed_k{k}" in res:
        axes[0].plot(x, res[f"jac_seed_k{k}"].mean(1), "k--", lw=1.2, label="same loop, other init seed")
    for ax, t in zip(axes, ["Jaccard(top-k, ref loop)", "Jaccard(top-k, previous loop)", "unit-usage overlap with ref",
                            "#units emerging (solid) / vanishing (dashed)"]):
        ax.axvline(ref + 1, color="r", lw=0.8)
        ax.set_title(f"{args.model}: {t}, k={k}", fontsize=9)
        ax.set_xlabel("loop")
        if L > 16:
            ax.set_xscale("log")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    (out / "figs").mkdir(exist_ok=True)
    fig.savefig(out / "figs" / f"support_check_gsm8k_L{L}.png", dpi=120)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
