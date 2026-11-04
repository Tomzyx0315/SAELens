"""Feedforward baseline (Llama-3.2-1B): stage-of-inference metrics on the same 256 examples, averaged.

Writes results/llama/llama_baseline[_<data>].json, consumed by analyze.py as the red dashed line.
"""

import json
import sys

import numpy as np

from metrics import attn_metrics, load_texts, residual_stats
from models import get_adapter
from paths import results_dir

data = sys.argv[1] if len(sys.argv) > 1 else "gsm8k"
ad = get_adapter("llama")
acc: dict[str, list] = {}
for text in load_texts(data, 256):
    tr = ad.run(ad.encode(text))
    m = attn_metrics(tr.attn.float())
    m.update(residual_stats(tr.resid))
    for k, v in m.items():
        acc.setdefault(k, []).append(v.cpu().numpy())
res = {k: np.mean(v, 0).tolist() for k, v in acc.items()}
suffix = "" if data == "gsm8k" else f"_{data}"
out = results_dir("llama") / f"llama_baseline{suffix}.json"
out.parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(out, "w"))
np.set_printoptions(precision=3, suppress=True, linewidth=220)
for k, v in res.items():
    print(k, np.array(v))
