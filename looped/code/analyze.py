"""Aggregate collected npz files and produce figures mirroring arXiv 2604.11791.

Usage: python analyze.py --model {ouro,huginn}. Reads RAW_ROOT/<model>/<run>/, writes results/<model>/.
The recurrent-block size (NL) is read from each run's args.json.
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paths import raw_dir, results_dir

# set in main() from --model
RAW = R = FIG = ""
NL = 24
TRAIN_LOOPS = 4
SUMMARY: dict = {}


def load(name):
    global NL, TRAIN_LOOPS
    files = sorted(glob.glob(f"{RAW}/{name}/ex*.npz"))
    if not files:
        return None
    meta = json.load(open(f"{RAW}/{name}/args.json"))
    st = meta.get("structure", {})  # old Ouro runs have no structure field
    NL = st.get("n_core", 24)
    TRAIN_LOOPS = st.get("train_loops", 4)
    data = [dict(np.load(f)) for f in files]
    keys = [k for k in data[0] if not k.startswith("last_tok")]
    out = {k: np.stack([d[k] for d in data]) for k in keys}
    for k in ("last_tok_resid", "last_tok_state", "last_tok_normed"):
        if k in data[0]:
            out[k] = [d[k] for d in data]
    out["n"] = len(files)
    return out


def by_loop(x, loops):  # (n, loops*NL) -> (n, loops, NL)
    return x.reshape(x.shape[0], loops, NL)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}.png", dpi=130)
    plt.close(fig)


def heatmaps(d, loops, tag):
    """Fig 2 (attention Frobenius) and Fig 19 (residual cosine), depth x depth."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, k, title, cmap in [
        (axes[0], "pair_attn", "attention ||A_i - A_j||_F (mean over heads)", "viridis_r"),
        (axes[1], "pair_cos", "residual cosine similarity", "viridis"),
    ]:
        m = d[k].mean(0)
        im = ax.imshow(m, cmap=cmap)
        ax.set_title(f"{tag}: {title}")
        for b in range(1, loops):
            ax.axhline(b * NL - 0.5, color="w", lw=0.4)
            ax.axvline(b * NL - 0.5, color="w", lw=0.4)
        plt.colorbar(im, ax=ax, fraction=0.046)
    savefig(fig, f"fig2_19_heatmaps_{tag}")

    # quantify "same layer across loops is most similar": for each depth (loop>=1), is the nearest other depth
    # (by attention distance) the same layer index in another loop?
    m = d["pair_attn"].mean(0)
    D_ = m.shape[0]
    hits, hits_c = 0, 0
    mc = d["pair_cos"].mean(0)
    for i in range(NL, D_):
        row = m[i].copy()
        row[i] = np.inf
        hits += (row.argmin() % NL) == (i % NL)
        rc = mc[i].copy()
        rc[i] = -np.inf
        hits_c += (rc.argmax() % NL) == (i % NL)
    SUMMARY[f"{tag}_nearest_attn_is_same_layer_frac"] = hits / (D_ - NL)
    SUMMARY[f"{tag}_nearest_resid_is_same_layer_frac"] = hits_c / (D_ - NL)
    # same layer, consecutive loops vs. adjacent layer within loop
    same_layer = np.mean([m[t * NL + l, (t + 1) * NL + l] for t in range(1, loops - 1) for l in range(NL)])
    adj_layer = np.mean([m[t * NL + l, t * NL + l + 1] for t in range(1, loops) for l in range(NL - 1)])
    SUMMARY[f"{tag}_attnF_same_layer_next_loop(t>=1)"] = float(same_layer)
    SUMMARY[f"{tag}_attnF_adjacent_layer_same_loop"] = float(adj_layer)


def succ_and_fp(d, loops, tag):
    """Fig 3/24/25 (successive) and Fig 4/22/23 (distance to last loop)."""
    cm = plt.get_cmap("winter")
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    names = [("succ_norm", "||x_t - x_{t-1}||"), ("succ_cos", "cos(x_t, x_{t-1})"), ("succ_attn", "||A_t - A_{t-1}||_F"),
             ("fp_norm", f"||x_t - x_{loops}||"), ("fp_cos", f"cos(x_t, x_{loops})"), ("fp_attn", f"||A_t - A_{loops}||_F")]
    for ax, (k, title) in zip(axes.flat, names):
        m = d[k].mean(0)  # (loops, NL)
        t0 = 1 if k.startswith("succ") else 0
        t1 = loops if k.startswith("succ") else loops - 1
        x = np.arange(t0 + 1, t1 + 1)
        for l in range(NL):
            ax.plot(x, m[t0:t1, l], color=cm(l / (NL - 1)), lw=1)
        if loops > 16:
            ax.set_xscale("log")
        ax.set_title(f"{tag}: {title}")
        ax.set_xlabel("recurrence")
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, NL - 1))
    fig.colorbar(sm, ax=axes, label="layer", fraction=0.02)
    fig.savefig(f"{FIG}/fig3_4_22_25_dynamics_{tag}.png", dpi=130)
    plt.close(fig)
    SUMMARY[f"{tag}_succ_cos_by_loop(mean over layers)"] = d["succ_cos"].mean((0, 2))[1:].round(4).tolist()
    SUMMARY[f"{tag}_fp_cos_by_loop(mean over layers)"] = d["fp_cos"].mean((0, 2)).round(4).tolist()


def stages(ds, tag, baseline=None):
    """Fig 8/35: stage metrics per loop over % depth within the loop. ds: list of (label, d, loops)."""
    metrics = ["colsum_conc", "sink_rate", "mixing_norm", "resid_entropy", "resid_norm", "resid_norm_excl0"]
    for label, d, loops in ds:
        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.6))
        cm = plt.get_cmap("summer")
        x = np.arange(NL) / (NL - 1) * 100
        for ax, k in zip(axes, metrics):
            v = by_loop(d[k], loops)
            mu = v.mean(0)
            for t in range(loops):
                ax.plot(x, mu[t], color=cm(t / max(loops - 1, 1)), lw=1.3, label=f"loop {t+1}" if loops <= 8 else None)
            if baseline is not None and k in baseline:
                bx = np.arange(len(baseline[k])) / (len(baseline[k]) - 1) * 100
                ax.plot(bx, baseline[k], "r--", lw=1.3, label="Llama-3.2-1B")
            ax.set_title(k)
            ax.set_xlabel("% depth in loop")
        if loops <= 8 or baseline is not None:
            axes[0].legend(fontsize=7)
        fig.suptitle(f"{tag} {label}: stages of inference per loop (mean over n={d['n']})")
        savefig(fig, f"fig8_35_stages_{tag}_{label}")


def realized_depth(d, loops, tag):
    """Fig 34 / Fig 9: metrics over realized depth (no grouping)."""
    metrics = ["colsum_conc", "sink_rate", "mixing_norm", "resid_entropy", "resid_norm"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 2.4 * len(metrics)), sharex=True)
    for ax, k in zip(axes, metrics):
        mu, sd = d[k].mean(0), d[k].std(0)
        x = np.arange(len(mu))
        ax.plot(x, mu, "k-", lw=1)
        ax.fill_between(x, mu - sd, mu + sd, alpha=0.2)
        for b in range(1, loops):
            ax.axvline(b * NL - 0.5, color="gray", lw=0.5, ls=":")
        ax.set_ylabel(k)
    axes[-1].set_xlabel("realized depth (layer index across loops)")
    fig.suptitle(f"{tag}: metrics over realized depth")
    savefig(fig, f"fig9_34_realized_depth_{tag}")


def stability(d, loops, tag):
    """Fig 11 / 47 (metric per layer vs recurrence) and Fig 12 / 50 (metric vs % depth for every recurrence)."""
    metrics = ["colsum_conc", "sink_rate", "mixing_norm", "resid_entropy"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 3.8))
    cm = plt.get_cmap("winter")
    for ax, k in zip(axes, metrics):
        v = by_loop(d[k], loops).mean(0)
        for l in range(NL):
            ax.plot(np.arange(1, loops + 1), v[:, l], color=cm(l / (NL - 1)), lw=0.9)
        ax.axvline(TRAIN_LOOPS, color="r", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_title(k)
        ax.set_xlabel(f"recurrence (red: train depth {TRAIN_LOOPS})")
    savefig(fig, f"fig11_47_stability_by_layer_{tag}")

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 3.8))
    cm = plt.get_cmap("summer")
    x = np.arange(NL) / (NL - 1) * 100
    for ax, k in zip(axes, metrics):
        v = by_loop(d[k], loops).mean(0)
        for t in range(loops):
            ax.plot(x, v[t], "-o" if NL <= 4 else "-", ms=2, color=cm(t / (loops - 1)), lw=0.6, alpha=0.8)
        ax.plot(x, v[TRAIN_LOOPS - 1], "k--", lw=1.4, label=f"loop {TRAIN_LOOPS} (train depth)")
        ax.set_title(k)
        ax.set_xlabel("% depth in loop")
    axes[0].legend(fontsize=7)
    savefig(fig, f"fig12_50_stability_by_depth_{tag}")

    cs = by_loop(d["colsum_conc"], loops).mean(0)
    SUMMARY[f"{tag}_colsum_layer_mean_abs_change_last_half"] = float(np.abs(np.diff(cs[loops // 2 :], axis=0)).mean())
    ref = TRAIN_LOOPS - 1
    SUMMARY[f"{tag}_colsum_dev_from_train_depth_loop{TRAIN_LOOPS}_at_loops"] = {
        str(t): float(np.abs(cs[t - 1] - cs[ref]).mean()) for t in (2, 4, 8, 16, 32, 64, 128) if t <= loops
    }


def pca(name, loops):
    d = load(name)
    if d is None:
        return
    X = d["last_tok_resid"][0]  # (loops*NL, D)
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    Z = (X - mu) @ Vt[:2].T
    Z = Z.reshape(loops, NL, 2)
    q = [(0, loops // 4), (loops // 4, loops // 2), (loops // 2, 3 * loops // 4), (3 * loops // 4, loops)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharex=True, sharey=True)
    cm = plt.get_cmap("summer")
    for ax, (a, b) in zip(axes, q):
        for t in range(loops):
            ax.plot(Z[t, :, 0], Z[t, :, 1], "-", color="lightgray", lw=0.4)
        for t in range(a, b):
            ax.plot(Z[t, :, 0], Z[t, :, 1], "-o", ms=2, color=cm((t - a) / max(b - a - 1, 1)), lw=0.8)
        ax.set_title(f"recurrences {a+1}-{b}")
    fig.suptitle(f"PCA of last-token residual after every layer: {name} (var expl {S[:2]**2 / (S**2).sum()})")
    savefig(fig, f"fig26_28_pca_{name}")


def geometry(tag):
    """Raw vs centered cosine vs linear CKA heatmaps over depth (from geometry.py) + same-layer summary."""
    f = f"{R}/geometry_{tag}.npz"
    if not os.path.exists(f):
        return
    g = np.load(f)
    nl = NL
    D_ = g["cka"].shape[0]
    loops = D_ // nl
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax, k, title in [(axes[0], "raw_cos", "raw cosine"), (axes[1], "cen_cos", "centered cosine"),
                         (axes[2], "cka", "linear CKA")]:
        im = ax.imshow(g[k], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"{tag}: {title} (N={int(g['N'])} tokens)")
        for b in range(1, loops):
            ax.axhline(b * nl - 0.5, color="w", lw=0.4)
            ax.axvline(b * nl - 0.5, color="w", lw=0.4)
        plt.colorbar(im, ax=ax, fraction=0.046)
    savefig(fig, f"geometry_{tag}")
    for k in ("raw_cos", "cen_cos", "cka"):
        m = g[k]
        hits = 0
        for i in range(nl, D_):
            r = m[i].copy()
            r[i] = -np.inf
            hits += (r.argmax() % nl) == (i % nl)
        SUMMARY[f"geometry_{tag}_{k}"] = {
            "same_layer_next_loop(t>=2)": float(np.mean([m[t * nl + l, (t + 1) * nl + l] for t in range(1, loops - 1) for l in range(nl)])),
            "adjacent_layer_same_loop(t>=2)": float(np.mean([m[t * nl + l, t * nl + l + 1] for t in range(1, loops) for l in range(nl - 1)])),
            "loop1_vs_loop2_same_layer": float(np.mean([m[l, nl + l] for l in range(nl)])),
            "nearest_is_same_layer_frac(t>=2)": hits / (D_ - nl),
        }
    SUMMARY[f"geometry_{tag}_mean_energy_frac_by_loop"] = g["mean_energy_frac"].reshape(loops, nl).mean(1).round(3).tolist()


def sinks(d, loops, tag):
    """Where is the attention sink: per (loop, layer), sink rate, mass on position 0, top-key==pos0, massive ratio."""
    keys = [k for k in ("sink_rate", "sink_score", "top_key_is_pos0", "massive_ratio") if k in d]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    for ax, k in zip(np.atleast_1d(axes), keys):
        v = by_loop(d[k], loops).mean(0)  # (loops, NL)
        im = ax.imshow(v, aspect="auto", cmap="magma")
        ax.set_title(k)
        ax.set_xlabel("layer in block")
        ax.set_ylabel("loop")
        ax.set_yticks(range(loops))
        ax.set_yticklabels(range(1, loops + 1))
        plt.colorbar(im, ax=ax)
        SUMMARY[f"{tag}_{k}_by_loop(mean over layers)"] = v.mean(1).round(3).tolist()
    fig.suptitle(f"{tag}: attention sink at position 0 (no BOS for Ouro; BOS for Huginn)")
    savefig(fig, f"sinks_{tag}")


def main():
    global RAW, R, FIG
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ouro", choices=["ouro", "huginn"])
    args = p.parse_args()
    RAW, R = str(raw_dir(args.model)), str(results_dir(args.model))
    FIG = f"{R}/figs"
    os.makedirs(FIG, exist_ok=True)

    for loops, name in [(8, "gsm8k_L8"), (4, "gsm8k_L4"), (32, "gsm8k_L32"), (4, "hellaswag_L4"),
                        (32, "hellaswag_L32"), (32, "hello_L32"), (32, "gsm8k_L32_seed1000")]:
        d = load(name)
        if d is None:
            continue
        print(name, d["n"])
        if "pair_attn" in d:
            heatmaps(d, loops, name)
        succ_and_fp(d, loops, name)
        realized_depth(d, loops, name)
        if loops <= 32:
            sinks(d, loops, name)
        if "loop_ce" in d:
            SUMMARY[f"{name}_logit_lens_ce_per_loop"] = d["loop_ce"].mean(0).round(4).tolist()
        geometry(name)

    base = None
    base_path = f"{results_dir('llama')}/llama_baseline.json"
    if os.path.exists(base_path):
        base = json.load(open(base_path))
    for name, loops in [("gsm8k", 4), ("gsm8k", 8), ("gsm8k", 32), ("hellaswag", 4), ("hellaswag", 8), ("hellaswag", 32)]:
        d = load(f"{name}_L{loops}")
        if d is not None:
            stages([(f"L{loops}", d, loops)], name, baseline=base)

    for name in ["gsm8k_L128", "hello_L128", "sqrt16_L128"]:
        d = load(name)
        if d is not None and d["n"] > 0:
            print(name, d["n"])
            stability(d, 128, name)
            succ_and_fp(d, 128, name)
            if "loop_ce" in d:
                SUMMARY[f"{name}_logit_lens_ce_per_loop"] = d["loop_ce"].mean(0)[[0, 1, 2, 3, 5, 7, 15, 31, 63, 127]].round(4).tolist()
    for name, loops in [("hello_L32", 32), ("sqrt16_L32", 32), ("hello_L128", 128), ("sqrt16_L128", 128)]:
        pca(name, loops)

    # bf16 vs fp32: is the 128-loop drift a numerical artifact?
    fp, bf = load("gsm8k_L128_fp32"), load("gsm8k_L128")
    if fp is not None and bf is not None:
        n = fp["n"]
        T = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
        for nm, d in [("bf16", {k: v[:n] for k, v in bf.items() if isinstance(v, np.ndarray) and v.ndim > 0}), ("fp32", fp)]:
            SUMMARY[f"L128_{nm}_fp_cos_mean_layers@{T}"] = d["fp_cos"].mean((0, 2))[[t - 1 for t in T]].round(3).tolist()
            SUMMARY[f"L128_{nm}_succ_cos_mean_layers@{T}"] = d["succ_cos"].mean((0, 2))[[t - 1 for t in T]].round(3).tolist()
            SUMMARY[f"L128_{nm}_colsum_mean_layers@{T}"] = by_loop(d["colsum_conc"], 128).mean((0, 2))[[t - 1 for t in T]].round(3).tolist()
            SUMMARY[f"L128_{nm}_loop_ce@{T}"] = d["loop_ce"].mean(0)[[t - 1 for t in T]].round(3).tolist()
        stability(fp, 128, "gsm8k_L128_fp32")
        succ_and_fp(fp, 128, "gsm8k_L128_fp32")
    fp8 = load("gsm8k_L8_fp32")
    b8 = load("gsm8k_L8")
    if fp8 is not None and b8 is not None:
        for k in ("colsum_conc", "sink_rate", "mixing_norm"):
            SUMMARY[f"L8_bf16_vs_fp32_max_abs_diff_{k}"] = float(np.abs(fp8[k].mean(0) - b8[k].mean(0)).max())
        SUMMARY["L8_fp32_logit_lens_ce_per_loop"] = fp8["loop_ce"].mean(0).round(4).tolist()

    # correlation between Ouro per-loop stage profile and Llama profile (resampled to same % depth).
    # Only meaningful for Ouro (24-layer block); Huginn's block has 4 layers.
    d4 = load("gsm8k_L4") if args.model == "ouro" else None
    if base is not None and d4 is not None:
        for k in ("colsum_conc", "sink_rate", "mixing_norm", "resid_entropy"):
            b = np.interp(np.linspace(0, 1, NL), np.linspace(0, 1, len(base[k])), base[k])
            v = by_loop(d4[k], 4).mean(0)
            SUMMARY[f"pearson_ouroL4_vs_llama_{k}_per_loop"] = [round(float(np.corrcoef(v[t], b)[0, 1]), 3) for t in range(4)]

    with open(f"{R}/summary.json", "w") as f:
        json.dump(SUMMARY, f, indent=1, default=float)
    print(json.dumps(SUMMARY, indent=1, default=float))


if __name__ == "__main__":
    main()
