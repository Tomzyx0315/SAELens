"""Model-agnostic data loading and per-depth metrics (attention stages of inference, residual statistics)."""

import math
import random

import pandas as pd
import torch
import torch.nn.functional as F

from paths import DATA

SINK_TAU = 0.3

PROMPTS = {
    "hello": "Hello! I've been well. I hope that you're doing well.",
    "sqrt16": "The square root of 16 is",
}


def load_texts(name: str, n: int) -> list[str]:
    if name == "gsm8k":
        df = pd.read_parquet(DATA / "gsm8k/main/test-00000-of-00001.parquet")
        texts = df["question"].tolist()
    elif name == "hellaswag":
        df = pd.read_parquet(DATA / "hellaswag/data/test-00000-of-00001.parquet")
        texts = df["ctx"].tolist()
    elif name in PROMPTS:
        return [PROMPTS[name]]
    else:
        raise ValueError(name)
    idx = random.Random(0).sample(range(len(texts)), n)
    return [texts[i] for i in idx]


def attn_metrics(A: torch.Tensor, sink_pos: int = 0) -> dict[str, torch.Tensor]:
    """A: (depth, H, T, T) fp32 row-stochastic causal attention. Returns per-depth metrics (mean over heads).

    sink_pos is the key position treated as the sink (0 = first token; BOS if the tokenizer prepends one).
    """
    T = A.shape[-1]
    eps = 1e-12
    col = A.sum(-2) / T  # (depth, H, T), sums to 1
    colsum_conc = 1 + (col * (col + eps).log()).sum(-1) / math.log(T)
    sink_score = A[..., sink_pos].mean(-1)  # (depth, H)
    sink_rate = (sink_score >= SINK_TAU).float()
    row_ent = -(A * (A + eps).log()).sum(-1)  # (depth, H, T)
    # normalise each row's entropy by log(#allowed keys) = log(i+1); row 0 has entropy 0
    denom = torch.log(torch.arange(1, T + 1, device=A.device, dtype=A.dtype)).clamp(min=1e-6)
    mixing = row_ent.mean(-1)
    mixing_norm = (row_ent[..., 1:] / denom[1:]).mean(-1) if T > 1 else torch.zeros_like(mixing)
    return {
        "colsum_conc": colsum_conc.mean(-1),
        "sink_rate": sink_rate.mean(-1),
        "sink_score": sink_score.mean(-1),
        "mixing": mixing.mean(-1),
        "mixing_norm": mixing_norm.mean(-1),
        "top_key_is_pos0": (col.mean(1).argmax(-1) == 0).float(),  # head-averaged most-attended key is position 0
    }


def residual_entropy(X: torch.Tensor) -> torch.Tensor:
    """Matrix-based (alpha=1) entropy of each residual (depth, T, D), normalised by log T (Skean et al.)."""
    X = X.float()
    K = X @ X.transpose(-1, -2)
    K = K / K.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None]
    lam = torch.linalg.eigvalsh(K.double()).clamp(min=0).float()
    lam = lam / lam.sum(-1, keepdim=True)
    H = -(lam * (lam + 1e-12).log()).sum(-1)
    return H / math.log(X.shape[1]) if X.shape[1] > 1 else H


def residual_stats(R: torch.Tensor) -> dict[str, torch.Tensor]:
    """R: (depth, T, D). Norm statistics, incl. position-0 massive-activation ratio."""
    R = R.float()
    norms = R.norm(dim=-1)  # (depth, T)
    T = R.shape[1]
    out = {"resid_norm": norms.mean(-1), "resid_entropy": residual_entropy(R)}
    if T > 1:
        out["resid_norm_excl0"] = norms[:, 1:].mean(-1)
        out["massive_ratio"] = norms[:, 0] / norms[:, 1:].median(-1).values
    else:
        out["resid_norm_excl0"] = out["resid_norm"]
        out["massive_ratio"] = torch.ones_like(out["resid_norm"])
    return out


def tok_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=-1)
