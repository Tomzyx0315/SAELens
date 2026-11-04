# Datasets and model weights

All large artifacts live on `/data_fs/zhangyuxuan/looped/`; downloaded via the hf-mirror endpoint
(`HF_ENDPOINT=https://hf-mirror.com`). Provenance and exact subsets are recorded here.

## Model weights (`/data_fs/zhangyuxuan/looped/`)

| local dir | HF repo | revision / sha | size | notes |
|---|---|---|---|---|
| `Ouro-1.4B/` | `ByteDance/Ouro-1.4B` | model.safetensors sha256 58872a72…9885af | 2.9 GB | looped LM, (0,24,0), train depth 4, trust_remote_code |
| `huginn-0125/` | `tomg-group-umd/huginn-0125` | — | 15.6 GB (fp32) | looped LM, (2,4,2)+input injection, train depth ~32, trust_remote_code |
| `Llama-3.2-1B/` | `unsloth/Llama-3.2-1B` | — | 2.4 GB | feedforward baseline; same weights as meta-llama/Llama-3.2-1B, ungated mirror |

## Analysis / evaluation datasets (`/data_fs/zhangyuxuan/looped/data/`)

| local dir | HF repo | subset used | used for |
|---|---|---|---|
| `gsm8k/` | `openai/gsm8k` | `main/test`, 256 random rows (seed 0), `question` field only | stages / attention / geometry metrics; SAE feature interpretation |
| `hellaswag/` | `Rowan/hellaswag` | `data/test`, 256 random rows (seed 0), `ctx` field | non-reasoning control (paper Fig 42) |

## SAE training data (`/data_fs/zhangyuxuan/looped/data/huginn_val/`)

- **Source:** `tomg-group-umd/huginn-dataset`, the public record of Huginn-0125's pretraining data
  (~1.12T tokens total; data mix in its README: fineweb-edu, starcoder, dolma, proof-pile-2, open-web-math,
  MathPile, AutoMathText, etc.). We use the **validation** split only.
- **Downloaded subset:** 24 shards `validation/shard_10{00..09,0..23}_of_4096.parquet` (filename glob `shard_10*`),
  1.4 GB, 92,563 rows x 4097 tokens ≈ **379 M tokens**. (The full 1.06 TB dataset was NOT downloaded.)
- **Format:** each row is 4097 Huginn tokens (BOS-prefixed), with a human-readable `text` field. Rows may contain
  the training chat template `[65506, 72, 3125, 20359, 65507]` = `<|begin_header|>Huginn<|end_header|>`.
- **Why this dataset:** SAE features should cover what the model actually processes. Huginn's pretraining mix is
  heavily math/code/synthetic-textbook; a generic web corpus (fineweb) would undersample exactly the reasoning
  content we care about. Tokens are already produced by Huginn's own tokenizer. The validation split is held out
  from training, avoiding memorization confounds.
- **Reserved for interpretation only:** GSM8k and HellaSwag are used to read out / interpret SAE features, never
  to train them.

### SAE token-budget note
First SAE run targets ~5e7 text tokens -> with per-token sampling of 10 loops x 2 init seeds that is ~1e9
activation vectors. This may be too few (open question; flagged by us) — the downloaded 379 M text tokens leave
headroom to scale up by ~7x without downloading more. If more is needed, extend the `shard_10*` glob.

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
```

## SAE runs

| run | data | site | SAE | budget | status |
|---|---|---|---|---|---|
| `huginn_loopstate_bt32k_k32_v1` | huginn-dataset validation (24 shards above), context 512, BOS dropped | Huginn loop state (last core-layer output), d_in 5280 | BatchTopK, d_sae 32768, k 32 | 1e9 vectors = ~5e7 text tokens x loops {1,2,3,4,6,8,12,16,24,32} x 2 init seeds | queued (waits for 8 idle GPUs) |

Output: `/data_fs/zhangyuxuan/looped/sae/<run>/` (`final_*` training checkpoint, `inference/` JumpReLU export,
`feature_loop_tags.npz` per-latent firing frequency by loop and by init seed, `run_args.json` incl. held-out check).
Launcher: `looped/code/launch_sae.sh`; log `looped/results/logs/sae_<run>.log`.
