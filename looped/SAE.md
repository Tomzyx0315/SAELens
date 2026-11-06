# Cross-loop SAE on Huginn-0125

Goal: train one SAE shared across loops on Huginn's loop state, then use per-feature firing statistics by loop to
ask whether loops reuse the same features (and which features appear only early / only after convergence).
Data provenance: `DATA.md`. Reproduction of the looped-model paper: `REPORT.md`.

## Feasibility check before training (no SAE)

`code/support_check.py`, results `results/{huginn,ouro}/support_check_gsm8k_*.{json,npz}`,
figure `results/huginn/figs/support_check_gsm8k_L48.png`. Question: across loops, does the *set* of active units
change, or do the same units just get stronger? If the active set never changed, an SAE would have little to say
about loop dynamics. Basis: the MLP hidden activation of each core layer (Huginn 17920-d, Ouro 5632-d); per token the
top-k units by |h| (k = 32 and 128, same conclusions); Jaccard overlap with the reference loop (train depth: Huginn 32,
Ouro 4). 256 GSM8k questions (Huginn 14.5k tokens), position 0 excluded. Baselines at the reference loop: adjacent
tokens 0.07-0.09, distant tokens 0.015-0.018.

Huginn, k = 128, mean over the 4 core layers:

| loop | 1 | 2 | 4 | 8 | 16 | 32 | 48 |
|---|---|---|---|---|---|---|---|
| Jaccard with loop 32 | 0.06 | 0.11 | 0.25 | 0.52 | 0.82 | 1 | 0.89 |
| same loop, other init seed | 0.27 | 0.41 | 0.59 | 0.76 | 0.91 | 0.96 | 0.90 |
| population unit-usage overlap with loop 32 | 0.53 | 0.62 | 0.83 | 0.95 | 0.98 | 1 | 0.99 |
| units used at loop 32 but (almost) unused here | ~1100 | ~700 | ~28 | 0 | 0 | – | 0 |

- A token's active set at loop 1 overlaps the converged one less than two adjacent tokens do; it converges
  monotonically by loops 16-24. So the active set is replaced, not just re-weighted.
- Units entering / leaving the population are confined to loops 1-4; afterwards only per-token combinations change.
- Early-loop differences are partly init noise (seed-vs-seed Jaccard only 0.27 at loop 1), but the gap to the
  converged state (0.06) is well beyond it.
- The neuron basis is dense (top-128 units carry only 8.6% of |h|), so this is a proxy; it motivated learning a
  sparse basis (the SAE).
- Ouro for contrast: at loop 8 (2x its train depth) the overlap with loop 4 is only 0.44-0.53, below the successive
  loop-to-loop overlap (0.72-0.81): the support drifts as soon as Ouro leaves its train depth. One reason to prefer
  Huginn for the SAE study.

Design consequences taken over into the SAE: loops 1-32 with emphasis on 1-8 (log-spaced loops), two init seeds per
document to separate init noise from computation, and the loop state as the site.

## Pipeline (no activations on disk)

| piece | file | what it does |
|---|---|---|
| activation stream | `code/sae_data.py` | Huginn producers (one process per GPU) read huginn-dataset validation rows (context 512, BOS dropped), run 32 loops x 2 random init seeds, emit the loop state at loops {1,2,3,4,6,8,12,16,24,32} tagged (loop, seed); a 2M-vector shuffle buffer mixes documents/loops/seeds |
| site | `models/huginn.py::run_states` | output of the last core-block layer per loop (the latent passed to the next loop), d_in = 5280 |
| training | `code/train_sae.py` | SAELens `BatchTopKTrainingSAE` + `SAETrainer.step`; resumable checkpoints every 20M vectors; input-norm scale estimated once and folded at the end; exports JumpReLU inference model; tags features by loop/seed on a held-out stream |
| launcher | `code/launch_sae.sh`, `code/_sae_job.sh` | waits until every GPU has >= 12 GB free (GPUs are shared with an sglang server), restarts from the latest checkpoint on failure |
| sweep | `code/sweep_sae.py` | several SAEs trained on the *same* batches (one GPU each), evaluated on the fixed held-out cache |
| eval cache | `code/cache_eval.py` | 647k held-out loop-state vectors, see `DATA.md` |
| snapshot | `code/sae_snapshot.py` | per-loop analysis of a mid-training checkpoint |

Hardware constraint: each 4090 has ~13.3 GB free next to the sglang server. One SAE with d_sae 32768 peaks at
9.2 GB (independent of k); one Huginn producer needs ~8 GB; they cannot share a GPU. d_sae 65536 does not fit.
Batch size is the SAELens default 4096 (not reduced for memory).

## How we judge SAE training

| metric | meaning | where |
|---|---|---|
| FVU | sum ||x - x_hat||^2 / sum ||x - mean||^2 on held-out data; 1 - FVU = explained variance | sweep evals (overall and per loop) |
| per-loop FVU | FVU on each loop separately; flags a dictionary biased towards converged loops | sweep evals |
| L0 | mean number of active features per vector; fixed to k by BatchTopK in training; checked against the exported JumpReLU model (threshold must have converged) | end of `train_sae.py` |
| dead features | trainer: no firing for > 2000 steps; eval: frequency 0 on the eval subsample (small samples over-count rare features as dead) | sweep evals |
| loss / aux loss | MSE should keep decreasing; aux loss > 0 means dead features are being revived | training log |
| downstream (TODO) | splice the reconstruction back into Huginn at one loop, measure CE increase and whether the error is amplified or damped by later loops | after the full run |

Reference point: Huginn loop states are very high-dimensional (participation ratio ~1230; PCA needs 657 components
for 50% variance; rank-128 PCA leaves FVU 0.78, rank-512 leaves 0.55), norms are nearly constant (76.4 +- 0.02,
the block re-normalises), and loop identity explains only 1.4% of variance. Loop 1 states are lower-dimensional
(PR ~287 vs ~1200 for loops >= 4). An FVU of ~0.35-0.45 is therefore far better than linear compression at the
same budget.

## v1 run (stopped at 30%; superseded by v2 below)

`huginn_loopstate_bt32k_k32_v1`: BatchTopK, d_sae 32768, k 32, lr 3e-4 constant, no decay; target 1e9 vectors.
Stopped after 3.0e8 vectors: MSE plateaued from ~1e4 steps on. Snapshot at 2.8e8 vectors (`sae_snapshot.py`,
24 held-out docs; numbers in `results/sae/v1_snapshot_280M.json`):

- FVU ~0.47 at every loop (0.45-0.49).
- Feature usage overlap with loop 32: 0.21 (loop 1), 0.41 (2), 0.61 (3), 0.75 (4), 0.88 (6), 0.92 (8), >= 0.96 (12+).
  ~3500 features used at loop 32 are (almost) unused at loop 1; ~480 are used at loop 1 only. Turnover is
  confined to loops 1-6.
- Per-loop feature frequencies agree between the two init seeds (overlap 0.94-0.98, correlation 0.995-0.998), so
  which features early loops use is not init noise (individual tokens' feature sets at loop 1 still are noisy).
- Correction: the snapshot reported 38% dead features; the trainer's own statistics show only 1 feature without
  firing for > 2000 steps. The snapshot sample (119k vectors) was too small. The real problem was the FVU plateau.

## Config sweep (100M vectors per config, same data stream, held-out cache)

Round 1 (full eval results incl. per-loop FVU: `results/sae/sweep_r1_results.json`; checkpoints deleted),
d_sae 32768, decay = linear decay to 0 over the last 20% of steps:

| config | FVU @20M | 40M | 60M | 80M | 100M | dead (eval) |
|---|---|---|---|---|---|---|
| k32, lr 3e-4, const (= v1) | 0.496 | 0.480 | 0.478 | 0.474 | 0.475 | 3900 |
| k32, lr 1e-3, decay | 0.557 | 0.556 | 0.561 | 0.559 | 0.513 | 13132 |
| k64, lr 3e-4, decay | 0.449 | 0.435 | 0.433 | 0.429 | 0.401 | 2281 |
| k128, lr 3e-4, decay | 0.404 | 0.391 | 0.389 | 0.386 | 0.358 | 1945 |

Findings: (1) the final lr decay is worth ~0.03 FVU (the decay phase is 80M -> 100M; the constant-lr baseline does
not move there) and is what v1 lacked; (2) larger k gives lower FVU and all still improve; (3) lr 1e-3 kills half the
dictionary; (4) trainer-side dead features are 3-4 for every lr-3e-4 config; (5) per-loop FVU is balanced
(k128: 0.37 / 0.35 / 0.38 / 0.36 / 0.36 at loops 1/2/4/8/32).

Round 2 (`results/sae/sweep_r2_results.json`; checkpoints in `/data_fs/zhangyuxuan/looped/sae/sweep_r2/`):

| config | FVU @20M | 40M | 60M | 80M | 100M | dead (eval) | dead (trainer) | per-loop FVU 1/2/4/8/32 |
|---|---|---|---|---|---|---|---|---|
| k64, lr 3e-4, const | 0.448 | 0.435 | 0.433 | 0.429 | 0.430 | 3141 | 1 | 0.42 / 0.41 / 0.45 / 0.44 / 0.44 |
| k64, lr 1e-4, decay | 0.463 | 0.428 | 0.419 | 0.412 | 0.393 | 470 | 0 | 0.41 / 0.39 / 0.41 / 0.40 / 0.39 |
| k96, lr 3e-4, decay | 0.422 | 0.410 | 0.407 | 0.404 | 0.376 | 2021 | 2 | 0.39 / 0.37 / 0.39 / 0.38 / 0.38 |
| **k128, lr 1e-4, decay** | 0.414 | 0.385 | 0.376 | 0.370 | **0.352** | **86** | 0 | 0.37 / 0.35 / 0.37 / 0.36 / 0.35 |

Findings: (1) decay control confirmed: k64 lr 3e-4 const ends at 0.430 vs 0.401 with decay (identical up to 80M,
diverging only in the decay phase); (2) lr 1e-4 beats 3e-4 at the same k (k64: 0.393 vs 0.401; k128: 0.352 vs 0.358):
slower start but keeps improving, and it nearly removes eval-dead features (86 vs 1945 at k128) - with 3e-4, some
features drift to near-zero frequency; (3) k is monotone (k64 < k96 < k128 in FVU), with no sign of the larger k
being worse on per-loop balance; (4) all configs still improving at 80M, so 1e9 vectors should help further.

**Chosen config for the full run:** BatchTopK, d_sae 32768, **k 128, lr 1e-4, constant + linear decay over the last
20% of steps**, no warmup, batch 4096 (SAELens default), dead_feature_window 1000 (SAELens default).
k was not pushed beyond 128: it is the largest value tested, and larger k makes individual features less
interpretable (the FVU gain per doubling of k shrinks: k64 -> k128 gives 0.04).

## Full run v2 (done)

`huginn_loopstate_bt32k_k128_v2`: chosen config above, 1e9 vectors (~5e7 text tokens x 10 loops x 2 seeds),
23 training shards (held-out shard excluded), feature tagging on the held-out shard. GPU 0 trained the SAE,
GPUs 1-7 ran Huginn producers, sharing GPUs with an sglang server. ~19 h wall-clock (one auto-resumed OOM restart).
Checkpoint `final_1000001536/`, JumpReLU export `inference/`, `feature_loop_tags.npz`, all under
`/data_fs/zhangyuxuan/looped/sae/huginn_loopstate_bt32k_k128_v2/`.

**Final metrics (held-out shard):** FVU **0.322** (train-mode) / 0.322 (inference JumpReLU); L0 128 / 127.6
(threshold converged, inference model is faithful); 64 dead features of 32768. Big improvement on v1's 0.47 plateau.

### Mid-training snapshot at 5.6e8 vectors (56%)

`sae_snapshot.py` on the `resume_560001024` checkpoint, 24 held-out docs:

| loop | 1 | 2 | 3 | 4 | 6 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|---|
| FVU | 0.368 | 0.352 | 0.371 | 0.377 | 0.372 | 0.365 | 0.356 | 0.356 |
| feature-usage overlap with loop 32 | 0.26 | 0.45 | 0.65 | 0.78 | 0.91 | 0.94 | 0.98 | 1.00 |
| overlap seed0 vs seed1 | 0.95 | 0.95 | 0.95 | 0.96 | 0.96 | 0.97 | 0.98 | 0.99 |
| features used here but ~unused at L32 | 356 | 374 | 267 | 142 | 26 | 5 | 0 | 0 |
| features used at L32 but ~unused here | 3389 | 1218 | 337 | 101 | 6 | 1 | 0 | 0 |

Overall FVU 0.358 (vs v1's ~0.47 plateau). Confirms the feasibility-check picture on learned sparse features:
feature turnover is confined to loops 1-4 and converges by loop 8; ~3400 features are used at the converged loop
but essentially unused at loop 1 ("late-emerging"), ~350 are used only at loop 1 ("early"). It is not init noise:
per-loop feature usage agrees across the two init seeds (overlap 0.95-0.99).

### Feature interpretability (mid-training)

`code/interpret_sae.py` collects each latent's max-activating token contexts across the per-domain eval corpora
(`DATA.md`) at a chosen loop. On the 560M checkpoint at loop 8 (160k tokens, 28287 live features;
`results/sae/interp_v2_560M_loop8.json`), many features are already cleanly interpretable, e.g.:

- 32134: " implied" in software-license boilerplate ("express or implied"), act ~40, freq 0.0004
- 555: fraction / percentage words ("one-third", "20 percent", "two-thirds")
- 18361: time-passing phrases ("Over time", "Over the years")
- 28250: emphatic adverbs ("particularly", "especially")
- 29853: the identifier "context" in Python (contextlib, context_processors)
- 12598: the builtin `len(`
- 1109: ".0" in floats

## Feature dynamics across loops (final SAE)

`feature_loop_tags.npz` (firing frequency per latent by loop and init seed, held-out shard). With
early = freq@loop1 > 1e-3 AND freq@loop32 < freq@loop1/5; late-emerging = symmetric; persistent = active at both:

- **early 1741, late-emerging 11940, persistent 6029** (of 32768; 64 fully dead). So the recurrence mostly *adds*
  features: ~12k latents only become active after the first loops, ~1.7k are used only early.
- Features active at loop 1 grow from 9447 (loop 1) to ~21000 (loop 3+). seed0-vs-seed1 firing-frequency
  correlation is 1.00 over live features: the per-loop feature usage is fully determined by computation, not by the
  random initial state.
- Early features fire more broadly (median freq@1 0.019, mean 0.044) than late features (median freq@32 0.005,
  mean 0.008): early loops use a small set of high-frequency, generic features; later loops add many sparse,
  specific ones.

### Early vs late-emerging semantics (`code/interpret_feats.py`, final SAE, 300 features each)

Interpreting the actual early set at loop 1 and the late set at loop 32 (max-activating contexts over the per-domain
corpora; `results/sae/interp_feats_early_late.json`):

- **Early (loop 1):** generic, high-frequency, often syntactic/positional features — common function words and
  punctuation in context ("[[ look]] into", "[[ It]] is", "https[[://]]", answer markers "[[Answer]]:"), firing on a
  large fraction of tokens. Domain mix 29% web / 30% code / 17% math.
- **Late-emerging (loop 32):** sparser, more specific features — particular code tokens (`.txt'`, `body_[[query]]`,
  `name`Last, license boilerplate "subject[[ to]]"), specific lexical/semantic items, numeric-result tokens in math
  ("<<300-12[[=]]288>>"). Domain mix 33% web / 28% code; slightly more web-weighted than early.
- Both classes are ~40% single-domain in their top contexts; the difference is firing breadth and specificity,
  not domain exclusivity.

Interpretation: Huginn's first loop sets up a generic, dense scaffold (syntax, position, frequent tokens); later
loops progressively switch on many sparse, content-specific features. This matches the geometry (loop 1 is
lower-dimensional, PR ~287 vs ~1200) and the stages-of-inference view.

### Max-activating examples (final SAE, loops 1/8/32)

`code/interpret_loops.py` interprets a sampled set of features at several loops in one forward pass per doc
(`results/sae/interp_final_loops.json`), used for spot-checking and the early/late example contexts above.

Still to do (TODO list): feature *flow* within a single sentence — track how features switch on/off across loops
(same layer and across layers) along the recurrent trajectory; and splicing the SAE reconstruction back into Huginn
to measure the CE increase and how the error propagates through loops.
