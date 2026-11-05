"""Build the per-domain SAE evaluation corpora (interpretation / per-domain loop analysis; never used for training).

Downloads (via hf-mirror, run in tmux) are done by `prep_domain_eval.sh`; this script turns the raw files into
one parquet per domain with Huginn token ids, so all downstream code reads a uniform format:

    /data_fs/zhangyuxuan/looped/data/domain_eval/<domain>.parquet   columns: doc_id, source, text, input_ids

- web:  HuggingFaceFW/fineweb-edu sample/10BT, last (smallest) shard 013_00000.parquet
- code: codeparrot/codeparrot-clean-valid (python files)
- math: EleutherAI/hendrycks_math test split, problem + solution (all 7 subjects); solution tokens are tagged
        via `answer_start` (token index where the solution begins) so reasoning tokens can be analysed separately
- gsm8k_full: openai/gsm8k test, question + answer (already on disk), with `answer_start`
- wiki: wikimedia/wikipedia 20231101.en, smallest shard

Every domain: N_DOCS documents sampled with a fixed seed, truncated to MAX_TOKENS Huginn tokens, BOS prepended
(Huginn tokenizer default), documents shorter than MIN_TOKENS dropped.
"""

import glob
import gzip
import json
import random

import pandas as pd
from transformers import AutoTokenizer

from paths import DATA, WEIGHTS

RAW = DATA / "domain_raw"
OUT = DATA / "domain_eval"
N_DOCS = 512
MAX_TOKENS = 512
MIN_TOKENS = 32
SEED = 0


def sample(rows, n):
    rows = list(rows)
    random.Random(SEED).shuffle(rows)
    return rows[: 4 * n]  # oversample; short docs are dropped after tokenization


def load_web():
    df = pd.read_parquet(glob.glob(str(RAW / "fineweb-edu/sample/10BT/013_00000.parquet"))[0], columns=["text"])
    return [{"text": t, "source": "fineweb-edu"} for t in sample(df["text"].tolist(), N_DOCS)]


def load_code():
    rows = []
    for f in sorted(glob.glob(str(RAW / "codeparrot-valid/*.json.gz"))):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                rows.append(json.loads(line)["content"])
    return [{"text": t, "source": "codeparrot-clean-valid"} for t in sample(rows, N_DOCS)]


def load_math():
    rows = []
    for f in sorted(glob.glob(str(RAW / "hendrycks_math/*/test-*.parquet"))):
        subj = f.split("/")[-2]
        df = pd.read_parquet(f)
        for p, s in zip(df["problem"], df["solution"]):
            rows.append({"prompt": f"Problem: {p}\nSolution:", "answer": f" {s}", "source": f"hendrycks_math/{subj}"})
    return sample(rows, N_DOCS)


def load_gsm8k():
    df = pd.read_parquet(DATA / "gsm8k/main/test-00000-of-00001.parquet")
    rows = [{"prompt": f"Question: {q}\nAnswer:", "answer": f" {a}", "source": "gsm8k/test"}
            for q, a in zip(df["question"], df["answer"])]
    return sample(rows, N_DOCS)


def load_wiki():
    f = sorted(glob.glob(str(RAW / "wikipedia/*.parquet")))[0]
    df = pd.read_parquet(f, columns=["text"])
    return [{"text": t, "source": "wikipedia/20231101.en"} for t in sample(df["text"].tolist(), N_DOCS)]


def tokenize(tok, rows):
    out = []
    for r in rows:
        if "prompt" in r:  # prompt/answer docs: record where the answer (reasoning) tokens start
            p = tok(r["prompt"]).input_ids  # BOS-prefixed
            a = tok(r["answer"], add_special_tokens=False).input_ids
            ids, text, answer_start = (p + a)[:MAX_TOKENS], r["prompt"] + r["answer"], len(p)
        else:
            ids, text, answer_start = tok(r["text"]).input_ids[:MAX_TOKENS], r["text"], -1
        if len(ids) < MIN_TOKENS or (answer_start >= 0 and answer_start >= len(ids) - 4):
            continue
        out.append({"doc_id": len(out), "source": r["source"], "text": text, "input_ids": ids,
                    "answer_start": answer_start})
        if len(out) == N_DOCS:
            break
    return out


def main():
    tok = AutoTokenizer.from_pretrained(WEIGHTS["huginn"])
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, loader in [("web", load_web), ("code", load_code), ("math", load_math), ("gsm8k_full", load_gsm8k),
                         ("wiki", load_wiki)]:
        rows = tokenize(tok, loader())
        df = pd.DataFrame(rows)
        df.to_parquet(OUT / f"{name}.parquet")
        n_tok = int(df["input_ids"].map(len).sum())
        summary[name] = {"docs": len(df), "tokens": n_tok,
                         "answer_tokens": int(sum(len(i) - s for i, s in zip(df["input_ids"], df["answer_start"]) if s >= 0)),
                         "sources": df["source"].value_counts().to_dict()}
        print(name, summary[name]["docs"], "docs", n_tok, "tokens", flush=True)
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
