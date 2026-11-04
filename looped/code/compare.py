"""Cross-model comparison figures (results/compare/):

- Fig 8/35 analogue: per-loop stage-of-inference metrics vs % depth within the recurrent block, each looped model
  next to the Llama-3.2-1B feedforward baseline (dashed red, resampled to the same % depth axis).
  Ouro is shown at its train depth (4 loops); Huginn at 8 loops (paper's setting) and its train mean 32.
- Fig 9 analogue: residual-stream norm over realized depth. For Huginn both the layer output (after norm_4)
  and the value before norm_4 are shown; prelude/coda layers are not captured (recurrent block only).

Caveat for Huginn: the paper's comparison is "recurrent block vs feedforward model"; Huginn has no feedforward
base model, so Llama-3.2-1B serves as the generic feedforward reference for both looped models.
"""

import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paths import RESULTS_ROOT, raw_dir, results_dir

OUT = RESULTS_ROOT / "compare"
OUT.mkdir(parents=True, exist_ok=True)
METRICS = ["colsum_conc", "sink_rate", "mixing_norm", "resid_entropy"]


def load(model, run):
    files = sorted(glob.glob(f"{raw_dir(model)}/{run}/ex*.npz"))
    meta = json.load(open(f"{raw_dir(model)}/{run}/args.json"))
    nl = meta.get("structure", {}).get("n_core", 24)
    data = [np.load(f) for f in files]
    keys = [k for k in data[0].files if not k.startswith("last_tok")]
    return {k: np.stack([d[k] for d in data]) for k in keys}, nl, meta["loops"]


def stage_curves(d, nl, loops, k):
    return d[k].reshape(d[k].shape[0], loops, nl).mean(0)  # (loops, nl)


def main():
    base = json.load(open(results_dir("llama") / "llama_baseline.json"))
    panels = [("Ouro-1.4B, 4 loops (train)", "ouro", "gsm8k_L4"),
              ("Huginn-0125, 8 loops", "huginn", "gsm8k_L8"),
              ("Huginn-0125, 32 loops (train mean)", "huginn", "gsm8k_L32")]
    summary = {}
    fig, axes = plt.subplots(len(panels), len(METRICS), figsize=(4.4 * len(METRICS), 3.3 * len(panels)))
    for r, (title, model, run) in enumerate(panels):
        d, nl, loops = load(model, run)
        x = np.arange(nl) / (nl - 1) * 100
        for c, k in enumerate(METRICS):
            ax = axes[r, c]
            v = stage_curves(d, nl, loops, k)
            cm = plt.get_cmap("summer")
            for t in range(loops):
                ax.plot(x, v[t], "-o" if nl <= 4 else "-", ms=3, color=cm(t / max(loops - 1, 1)), lw=1.2,
                        label=f"loop {t+1}" if loops <= 8 else None)
            bx = np.arange(len(base[k])) / (len(base[k]) - 1) * 100
            ax.plot(bx, base[k], "r--", lw=1.3, label="Llama-3.2-1B")
            ax.set_title(f"{title}\n{k}", fontsize=9)
            ax.set_xlabel("% depth in recurrent block")
            if c == 0 and loops <= 8:
                ax.legend(fontsize=6)
            # Pearson vs Llama after resampling Llama to nl points (only meaningful for nl > 4)
            b = np.interp(np.linspace(0, 1, nl), np.linspace(0, 1, len(base[k])), base[k])
            summary[f"{model}_{run}_{k}"] = {
                "range_within_loop(last loop)": float(np.ptp(v[-1])),
                "llama_range": float(np.ptp(base[k])),
                "pearson_vs_llama_per_loop": [round(float(np.corrcoef(v[t], b)[0, 1]), 3) for t in range(loops)]
                if nl > 4 else None,
            }
    fig.tight_layout()
    fig.savefig(OUT / "fig8_35_stages_vs_feedforward.png", dpi=120)
    plt.close(fig)

    # ---- Fig 9: residual norms over realized depth ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    lines = [("Ouro (layer output, unnormed residual)", "ouro", "gsm8k_L8", "resid_norm_excl0", "C0"),
             ("Huginn layer output (after norm_4)", "huginn", "gsm8k_L8", "resid_norm_excl0", "C1"),
             ("Huginn before norm_4", "huginn", "gsm8k_L8", "pre_resid_norm_excl0", "C2")]
    for label, model, run, k, col in lines:
        d, nl, loops = load(model, run)
        mu = d[k].mean(0)
        xd = (np.arange(len(mu)) + 1) / nl  # in units of loops
        axes[0].plot(xd, mu, color=col, lw=1.2, label=label)
        summary[f"fig9_{label}"] = {"min": float(mu.min()), "max": float(mu.max()), "max/min": float(mu.max() / mu.min())}
    lb = np.array(base["resid_norm_excl0"])
    axes[0].plot((np.arange(len(lb)) + 1) / len(lb), lb, "r--", lw=1.3, label="Llama-3.2-1B (one pass)")
    summary["fig9_llama"] = {"min": float(lb.min()), "max": float(lb.max()), "max/min": float(lb.max() / lb.min())}
    axes[0].set_xlabel("realized depth (in loops; Llama: one pass = 1)")
    axes[0].set_ylabel("mean ||x|| over non-initial tokens")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[0].set_title("residual norm (excluding position 0), 8 loops")

    # massive activation on position 0
    for label, model, run, k, col in [("Ouro", "ouro", "gsm8k_L8", "massive_ratio", "C0"),
                                      ("Huginn after norm_4", "huginn", "gsm8k_L8", "massive_ratio", "C1"),
                                      ("Huginn before norm_4", "huginn", "gsm8k_L8", "pre_massive_ratio", "C2")]:
        d, nl, loops = load(model, run)
        mu = d[k].mean(0)
        axes[1].plot((np.arange(len(mu)) + 1) / nl, mu, color=col, lw=1.2, label=label)
    axes[1].set_xlabel("realized depth (in loops)")
    axes[1].set_ylabel("||x_0|| / median_t>0 ||x_t||")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    axes[1].set_title("position-0 (sink token) norm ratio")
    fig.tight_layout()
    fig.savefig(OUT / "fig9_residual_norms.png", dpi=120)
    plt.close(fig)

    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main()
