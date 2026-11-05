# Datasets and model weights

All large artifacts live on `/data_fs/zhangyuxuan/looped/`; downloaded via the hf-mirror endpoint
(`HF_ENDPOINT=https://hf-mirror.com`). Provenance and exact subsets are recorded here.
SAE pipeline, metrics and sweep results are documented in `SAE.md`.

## Model weights (`/data_fs/zhangyuxuan/looped/`)

| local dir | HF repo | revision / sha | size | notes |
|---|---|---|---|---|
| `Ouro-1.4B/` | `ByteDance/Ouro-1.4B` | model.safetensors sha256 58872a72…9885af | 2.9 GB | looped LM, (0,24,0), train depth 4, trust_remote_code |
| `huginn-0125/` | `tomg-group-umd/huginn-0125` | — | 15.6 GB (fp32) | looped LM, (2,4,2)+input injection, train depth ~32, trust_remote_code |
| `Llama-3.2-1B/` | `unsloth/Llama-3.2-1B` | — | 2.4 GB | feedforward baseline; same weights as meta-llama/Llama-3.2-1B, ungated mirror |

## Paper-reproduction datasets (`/data_fs/zhangyuxuan/looped/data/`)

| local dir | HF repo | subset used | used for |
|---|---|---|---|
| `gsm8k/` | `openai/gsm8k` | `main/test`, 256 random rows (seed 0), `question` field only | stages / attention / geometry metrics (paper setting) |
| `hellaswag/` | `Rowan/hellaswag` | `data/test`, 256 random rows (seed 0), `ctx` field | non-reasoning control (paper Fig 42) |

These follow the paper's setup and are used only for the reproduction in `REPORT.md`. They are **not** the main
evaluation data for the SAE (GSM8k questions are a narrow, short, single-domain set; see below).

## SAE training data (`/data_fs/zhangyuxuan/looped/data/huginn_val/`)

- **Source:** `tomg-group-umd/huginn-dataset`, the public record of Huginn-0125's pretraining data
  (~1.12T tokens total; data mix in its README: fineweb-edu, starcoder, dolma, proof-pile-2, open-web-math,
  MathPile, AutoMathText, etc.). We use the **validation** split only.
- **Downloaded subset:** 24 shards `validation/shard_10{00..09,0..23}_of_4096.parquet` (filename glob `shard_10*`),
  1.4 GB, 92,563 rows x 4097 tokens ≈ **379 M tokens**. (The full 1.06 TB dataset was NOT downloaded.)
- **Train / held-out split:** `shard_1000_of_4096.parquet` is reserved for SAE evaluation and is never read during
  SAE training (`HELDOUT_SHARD` in `code/sae_data.py`); training uses the other 23 shards.
- **Format:** each row is 4097 Huginn tokens (BOS-prefixed), with a human-readable `text` field. Rows may contain
  the training chat template `[65506, 72, 3125, 20359, 65507]` = `<|begin_header|>Huginn<|end_header|>`.
  There is no per-row source/domain label.
- **Why this dataset:** SAE features should cover what the model actually processes. Huginn's pretraining mix is
  heavily math/code/synthetic-textbook; a generic web corpus (fineweb) would undersample exactly the reasoning
  content we care about. Tokens are already produced by Huginn's own tokenizer. The validation split is held out
  from Huginn's training, avoiding memorization confounds.

### SAE token-budget note
First full SAE run targets ~5e7 text tokens -> with per-token sampling of 10 loops x 2 init seeds that is ~1e9
activation vectors. Vectors of the same token at nearby (converged) loops are highly correlated, so the effective
data is closer to 5e7 tokens; this may be too few (open question). The downloaded 379 M text tokens leave headroom
to scale up ~7x without downloading more. If more is needed, extend the `shard_10*` glob.

## SAE evaluation data

### 1. Fixed held-out cache (main SAE metrics)
`/data_fs/zhangyuxuan/looped/sae/eval_cache.pt` (6.8 GB), built by `code/cache_eval.py`:
64 documents from the held-out shard (`shard_1000`), context 512, BOS dropped, loops {1,2,3,4,6,8,12,16,24,32}
x 2 init seeds -> 647,168 Huginn loop-state vectors (d 5280, fp16) with (loop, seed) tags. Same distribution as
the SAE training data but unseen. All sweep FVU / L0 / dead-feature numbers are computed on (a fixed 200k-vector
subsample of) this cache, so every config and training stage is directly comparable.

### 2. Per-domain evaluation corpora (interpretation and per-domain loop analysis)
`/data_fs/zhangyuxuan/looped/data/domain_eval/<domain>.parquet`, built by `code/prep_domain_eval.sh`
(download, tmux) + `code/prep_domain_eval.py` (sampling, tokenization). Columns: `doc_id, source, text, input_ids,
answer_start`. Each domain: up to 512 documents (seed 0), Huginn-tokenized with BOS, truncated to 512 tokens,
documents under 32 tokens dropped. Token counts in `domain_eval/summary.json`.

| domain | source (HF) | subset downloaded | notes |
|---|---|---|---|
| `web` | `HuggingFaceFW/fineweb-edu` | `sample/10BT/013_00000.parquet` (541 MB) | general educational web text, baseline |
| `code` | `codeparrot/codeparrot-clean-valid` | all (142 MB, python files) | structured text, long-range dependencies |
| `math` | `EleutherAI/hendrycks_math` | `*/test-*.parquet` (7 subjects) | `Problem: ... Solution: ...`; solution tokens marked by `answer_start` |
| `gsm8k_full` | `openai/gsm8k` (on disk) | `main/test` | question **and** worked answer; answer tokens marked by `answer_start` |
| `wiki` | `wikimedia/wikipedia` | `20231101.en/train-00028-of-00041.parquet` (188 MB, smallest shard) | encyclopedic / factual text |

Built corpora (512 docs each; tokens incl. BOS): web 223k, code 254k, math 132k (7 MATH subjects), gsm8k_full 94k,
wiki 142k. Raw downloads 0.83 GB; built parquets 5.8 MB.

- **Purpose:** test whether loop dynamics (early vs. late-emerging features, convergence speed) depend on domain,
  and interpret features on varied text. `answer_start` lets reasoning (solution) tokens be analysed separately
  from problem-statement tokens. Never used to train an SAE.
- **Caveat:** all five sources are (parts of) the families in Huginn's pretraining mix, so some documents may have
  been seen by Huginn during pretraining. This matters little for interpreting features; we use test/validation
  splits where available (math, gsm8k, codeparrot-valid).
- Raw downloads in `/data_fs/zhangyuxuan/looped/data/domain_raw/`.

## Download commands (for reproducibility)

```bash
export HF_ENDPOINT=https://hf-mirror.com
hf download ByteDance/Ouro-1.4B               --local-dir /data_fs/zhangyuxuan/looped/Ouro-1.4B
hf download tomg-group-umd/huginn-0125        --local-dir /data_fs/zhangyuxuan/looped/huginn-0125
hf download unsloth/Llama-3.2-1B              --local-dir /data_fs/zhangyuxuan/looped/Llama-3.2-1B --exclude 'original/*'
hf download openai/gsm8k     --repo-type dataset --local-dir /data_fs/zhangyuxuan/looped/data/gsm8k
hf download Rowan/hellaswag  --repo-type dataset --local-dir /data_fs/zhangyuxuan/looped/data/hellaswag
hf download tomg-group-umd/huginn-dataset --repo-type dataset \
    --include 'validation/shard_10*_of_4096.parquet' 'README.md' \
    --local-dir /data_fs/zhangyuxuan/looped/data/huginn_val
bash looped/code/prep_domain_eval.sh   # per-domain eval corpora (fineweb-edu, codeparrot, MATH, wikipedia)
```
