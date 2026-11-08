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

Caveat: these descriptions were written from hand-picked examples (features sorted by domain purity) and are
partly **not supported** by the quantitative test below: the syntactic-vs-content contrast does not hold.

### Quantitative descriptors of early / late / persistent features (`code/feature_descriptors.py`, A4)

LLM-free descriptors over all firings (328k tokens, 200 docs/domain; early at loop 1, late and persistent at
loop 32; `results/sae/feature_descriptors.json`). Medians; p = Mann-Whitney early vs late:

| descriptor | early (n=1735) | late (n=11936) | persistent (n=6021) | p |
|---|---|---|---|---|
| firing frequency | 0.022 | 0.0045 | 0.0031 | ~0 |
| domain entropy (1 = spread over all 5 domains) | **0.81** | **0.62** | 0.71 | 2e-143 |
| current-token entropy | 0.79 | 0.78 | 0.79 | 0.32 (n.s.) |
| share of top current token | 0.070 | 0.080 | 0.082 | 0.14 (n.s.) |
| punctuation share of firing tokens | 0.165 | 0.163 | 0.095 | 0.004 |
| mean relative position | 0.495 | 0.507 | 0.483 | 6e-15 |
| answer-token share (math / gsm8k) | 0.66 | 0.63 | 0.55 | 2e-11 |

What holds: early features fire ~5x more often and are much more **domain-general** (higher domain entropy);
late-emerging features are rarer and more **domain-specific**. What does not hold: early features are *not* more
token-specific / syntactic than late ones (token entropy and top-token share are indistinguishable; punctuation
shares are equal), and position effects are tiny. So the defensible summary is "early = few, frequent,
domain-general features; late = many, rare, domain-specific features", not "syntax first, content later".
Interpretability of individual features (does a feature correspond to a human concept?) still needs auto-interp,
which requires an LLM we have not set up.

### Max-activating examples (final SAE, loops 1/8/32)

`code/interpret_loops.py` interprets a sampled set of features at several loops in one forward pass per doc
(`results/sae/interp_final_loops.json`), used for spot-checking and the early/late example contexts above.

## Feature flow within a single sentence (`code/feature_flow.py`)

For a fixed prompt, each token has a (loops, d_sae) binary activation map on the loop state. We track how the active
feature set evolves along the recurrence, and relate it to the model's own per-loop logit-lens prediction.
Results `results/sae/flow_{math,fact,code}.json`.

**Feature turnover decays as the prediction settles.** Between consecutive loops, the fraction of active features
that stay on rises and new activations fall off, monotonically (math prompt, mean over tokens):

| loop step | 1->2 | 2->3 | 4->6 | 6->8 | 12->16 | 24->32 |
|---|---|---|---|---|---|---|
| frac of active features that stay on | 0.36 | 0.55 | 0.60 | 0.74 | 0.89 | 0.99 |
| newly-on features per token | 62 | 44 | 43 | 31 | 17 | 2 |

By loop 24->32 only ~1% of features change: the representation has essentially converged (consistent with the
fixed-point / stages results in `REPORT.md`). L0 is fixed at 128 throughout, so this is genuine turnover of *which*
features are on, not a change in sparsity.

**The feature flow tracks the forming prediction.** On "...currency is the yen is" the last-token logit-lens
prediction flips " the" -> " Tokyo" at loop 6 — exactly while feature turnover is still high (stay 0.6-0.7) — and
then stays " Tokyo" as turnover falls to ~1%. On "17 times 24 equals" the prediction moves 1 -> 4 around loops
12-16. So the recurrence is doing visible work in the first ~6-16 loops and mostly consolidating afterwards.

**Per (token, feature) activation classes** (math prompt): transient (on then off before the end) 40%, early-only
21%, late-only 18%, persistent 11%, intermittent 10%. Similar across prompts (fact / code). Most feature
activations are transient — features fire during the "computation" phase and switch off — rather than persisting to
the converged state. This is a token-level, dynamical complement to the population-level early/late split above.

Caveat: our SAE site is the end-of-loop state, so this is cross-loop flow at one site, not flow across the 4 core
layers within a loop (the loop state does not expose intermediate layers). Per-layer flow would need SAEs on the
core-layer outputs. The three single prompts above are anecdotes; the statistics are in the next section.

### Feature-flow statistics with baselines (`code/flow_stats.py`, A2)

1000 documents (200/domain), two init seeds each; mean with 95% bootstrap CI over documents (CIs are all
< +-0.004, omitted). Weighted turnover = 1 - sum min(a, a') / sum max(a, a') over feature activations (magnitude-
aware, insensitive to threshold flicker). Relative state distance = ||s_l' - s_l|| / ||s_l||.
`results/sae/flow_stats.json`.

| transition | stay (binary) | weighted turnover | state distance | turnover expected from isotropic noise of that size |
|---|---|---|---|---|
| 1 -> 2 | 0.31 | 0.81 | 0.91 | 0.61 |
| 2 -> 3 | 0.47 | 0.66 | 0.72 | 0.48 |
| 3 -> 4 | 0.57 | 0.55 | 0.59 | 0.40 |
| 6 -> 8 | 0.70 | 0.41 | 0.46 | 0.31 |
| 12 -> 16 | 0.87 | 0.20 | 0.19 | 0.13 |
| 24 -> 32 | 0.97 | 0.04 | 0.04 | 0.03 |

Baselines:
- **Same loop, other random init state:** turnover 0.50 at loop 1, 0.29 at loop 4, 0.08 at loop 12, 0.02 at loop 32.
  Init-seed variation is a large fraction of early-loop turnover: at loop 1 the two seeds differ in half of their
  feature mass. Consecutive-loop turnover is larger than seed turnover at every loop, so it is not only init noise.
- **Isotropic noise added to the loop-32 state:** turnover grows linearly, ~0.66 x relative noise norm.

Findings: (1) feature turnover tracks the size of the state change almost one-to-one (turnover ~ 0.9-1.1 x state
distance at every transition); the decay of turnover is just the convergence of the state, not an additional
discovery. (2) For the same state distance, real loop-to-loop changes turn over ~1.3-1.5x more feature mass than
isotropic noise: the trajectory moves along directions that the SAE dictionary resolves (it changes which features
are on), rather than in random directions. (3) The single-prompt class split (transient 40% etc.) uses a binary
threshold and has no flicker baseline; it should not be over-interpreted.

### Feature lineage: re-encoding or replacement? (`code/feature_lineage.py`, A3)

When features switch off between loops, do the features that switch on rebuild the same direction (the same
concept carried by another latent) or something else? 750 documents, 165k tokens. `results/sae/feature_lineage.json`.

| transition | features off (= on) per token | cos(v_off, v_on) per token | null (other token) | top-1 successor decoder cos (median) |
|---|---|---|---|---|
| 1 -> 2 | 88 | 0.125 | 0.057 | 0.008 |
| 2 -> 4 | 86 | 0.086 | 0.062 | -0.003 |
| 4 -> 8 | 66 | 0.065 | 0.070 | 0.001 |
| 8 -> 16 | 33 | 0.036 | 0.047 | 0.001 |

v_off = sum of decoder vectors (weighted by activation) of the features that switch off; v_on likewise for the
features that switch on. Decoder cosine of the most likely successor latent vs nearest-neighbour cosine in the whole
dictionary: median 0.18-0.23; fraction of top successors with decoder cos > 0.5: < 0.6%.

Finding: features that switch off are **not** replaced by latents pointing in the same direction. The direction
that leaves and the direction that arrives are nearly orthogonal (cos 0.04-0.13, barely above the other-token null,
and at null level from loop 4 on). So across loops the representation genuinely changes content in the SAE basis,
rather than re-encoding the same concept with different latents. Successors are only weakly predictable (top-1
successor probability median 0.08-0.51, lift 1.4-2.8 over its base rate). Limitation: this is a single shared
dictionary; the per-loop SAEs (B1) test the same question without assuming a shared basis.

## Splicing the SAE back into Huginn (`code/splice_eval.py`)

The ultimate test: replace the loop state with `sae.decode(sae.encode(state))` inside the forward pass and let the
recurrence continue. Next-token CE on the per-domain corpora (math/gsm8k: answer tokens only), 60 docs/domain,
32 loops. `results/sae/splice_final.json`.

| edit (one forward of 32 loops) | CE | delta CE | CE with mean-ablation at the same loop |
|---|---|---|---|
| clean (no edit) | 1.443 | - | - |
| replace **every** loop's state | 2.454 | +1.011 | - |
| replace only loop 1 / 2 / 4 / 8 | ~1.443 | ~0.000 | ~1.442 (also no effect) |
| replace only loop 16 | 1.4435 | +0.0006 | 1.445 |
| replace only loop 32 (last) | 1.732 | +0.289 | 7.15 -> loss recovered 0.95 |

**What this does and does not show (corrected).**

- At loops 1-16 even *mean-ablating* the state leaves CE unchanged. Huginn starts from a random state and converges
  to a fixed point, so any intermediate state is overwritten by the remaining loops. This single-edit test therefore
  has **no power** to measure SAE faithfulness at intermediate loops: zero delta CE there is a property of the model,
  not evidence for the SAE. (An earlier version of this section called it "self-correcting SAE error"; that
  overstates it.) The decaying error after injection (relative state error 0.58 at the edited loop -> 0.08 at
  loop 32) is the same contraction, not an SAE property.
- The only informative point is the read-out loop: editing loop 32 costs +0.29 CE, recovering 95% of the
  mean-ablation loss. Replacing every loop compounds to +1.01 CE.
- Faithfulness at intermediate loops is measured separately with depth-truncated runs (next section).

## Per-depth faithfulness (`code/depth_fidelity.py`, A1)

For each depth l, Huginn is run for exactly l loops and the loop-l state is replaced before being read by the
model's own coda + head, i.e. the model's real output at depth l. 500 documents (100/domain), math/gsm8k scored on
answer tokens. `results/sae/depth_fidelity.json`.

| depth | CE clean | CE SAE | CE mean-ablation | delta CE | KL(clean ‖ SAE) | loss recovered | FVU |
|---|---|---|---|---|---|---|---|
| 1 | 3.680 | 4.213 | 7.394 | +0.53 | 0.81 | 0.857 | 0.399 |
| 2 | 2.743 | 3.071 | 7.203 | +0.33 | 0.45 | 0.926 | 0.387 |
| 4 | 1.976 | 2.290 | 7.156 | +0.31 | 0.35 | 0.939 | 0.406 |
| 8 | 1.599 | 1.903 | 7.204 | +0.30 | 0.30 | 0.946 | 0.378 |
| 16 | 1.518 | 1.804 | 7.192 | +0.29 | 0.29 | 0.950 | 0.368 |
| 32 | 1.515 | 1.798 | 7.190 | +0.28 | 0.29 | 0.950 | 0.367 |

Per domain, delta CE at depth 1 / 4 / 32: web 0.41 / 0.28 / 0.26, code 0.58 / 0.33 / 0.30, math 0.47 / 0.15 / 0.13,
gsm8k 0.83 / 0.36 / 0.15, wiki 0.58 / 0.44 / 0.45.

Findings: the shared SAE is faithful at every depth (loss recovered 0.94-0.95 from depth 3 on), but **worst at
loop 1** (0.86, delta CE 0.53) even though its FVU there is no worse. Loop-1 states are read out by the model with
less tolerance to reconstruction error; one plausible reason is that the shared dictionary is dominated by the
~21k features of the converged regime while loop 1 uses only ~9k. Wiki stays hardest at every depth (delta CE ~0.45).
Loss recovered of 0.95 with delta CE 0.28 nat is typical of current residual-stream SAEs; it is the faithfulness
ceiling of all feature-level conclusions here.

## Is the representation space the same across loops? (B)

The analyses above use one shared dictionary, which presupposes a common space. B tests that assumption directly.

### B1. Per-loop SAEs and the cross-loop transfer matrix (`code/train_perloop_saes.py`)

One SAE per loop (BatchTopK, d_sae 16384, k 64, 60M vectors each), trained only on that loop's states, no shared
basis. Then FVU[train loop, eval loop] on the held-out cache. `results/sae/perloop_transfer.json`.

Transfer FVU (row = SAE's training loop, col = evaluation loop):

| train \ eval | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| **1** | 0.34 | 0.64 | 0.77 | 0.81 | 0.82 | 0.82 |
| **2** | 0.61 | 0.33 | 0.63 | 0.69 | 0.70 | 0.70 |
| **4** | 0.69 | 0.56 | 0.39 | 0.47 | 0.48 | 0.48 |
| **8** | 0.74 | 0.61 | 0.46 | 0.41 | 0.40 | 0.40 |
| **16** | 0.76 | 0.62 | 0.47 | 0.41 | 0.40 | 0.40 |
| **32** | 0.76 | 0.62 | 0.47 | 0.41 | 0.40 | 0.40 |

- **Loops >= 8 share one space.** The loop-8/16/32 SAEs are interchangeable: each reconstructs the others with FVU
  within 0.005 of the target loop's own SAE (e.g. loop-8 SAE on loop 32: 0.402 vs 0.398 own). Loops 8/16/32 are the
  same representation.
- **Early loops are different spaces.** Transferring to/from loop 1 costs ~0.42 excess FVU (loop-1 SAE on loop 32:
  0.82 vs 0.40; loop-32 SAE on loop 1: 0.76 vs 0.34) — i.e. a loop-32 dictionary explains loop-1 states hardly
  better than the mean. Loop 2 is intermediate (~0.28-0.30 excess), loop 4 closer (~0.07-0.08 to loops >= 8).
- **Decoder feature matching** (Hungarian on decoder cosine between consecutive loop SAEs) tells the same story:
  fraction of features with a matched counterpart at cos > 0.8 rises from 0.02 (1->2) to 0.05 (2->4), 0.16 (4->8),
  0.25 (8->16), 0.27 (16->32). Adjacent *late* loops share a quarter of their features as near-identical
  directions; adjacent *early* loops share almost none.

Answer to the original question: the loop representations are **not** a single shared space with features fading in
and out. There is a transition from loop 1 (a distinct, lower-dimensional space) through loops 2-4 to a stable
converged space that loops 8-32 all share. The "early / late-emerging" feature split from the shared SAE is the
shadow of this: the shared dictionary is really the converged-space dictionary, which fits loop 1 poorly.

### B2. Cross-loop crosscoder (`code/crosscoder.py`)

Shared latents, per-loop encoders/decoders, loops {1, 2, 4, 32} (Anthropic sparse-crosscoder setup across loops
instead of layers). Per-loop held-out FVU 0.40 / 0.39 / 0.46 / 0.50. Each latent has a decoder-norm share per loop;
a latent concentrated at loop 1 is early, spread across loops is shared. `results/sae/.../crosscoder results.json`.

- Of 6146 live latents, **0** are "mostly loop 1" (rel decoder norm > 0.7) and **0** are "mostly loop 32"; 4441 are
  spread across all four loops (each share 0.15-0.35). The relative-norm histograms peak at the 0.2-0.3 bin for both
  loop 1 and loop 32.
- Read with B1: the crosscoder ties loops together through shared latents and finds them mostly shared, but its
  loop-1 FVU (0.40) is no better than the shared SAE, and the per-loop decoders still differ. The crosscoder does
  not contradict B1; it shows that *if* you force shared latents, no latent is purely early/late — the early/late
  distinction lives in the encoder/decoder weights, not in a clean subset of latents.
- Caveat: this crosscoder underperformed its potential — ~2000 of 8192 latents died (the dead-latent aux-loss
  weight was mis-scaled, aux loss tracked the main loss). The decoder-norm *distribution* is still informative, but
  the absolute dead count should not be read as structure. A clean re-run is on the backlog.

### B3. Seed stability (`code/seed_stability.py`, `code/sweep_sae.py` seed runs)

Two SAEs with the main-run config (k128, lr 1e-4 + decay) differing only in parameter-init seed (100M vectors each),
compared with each other and with the 1B main run. `results/sae/seed_stability.json`.

- Both seeds reach identical FVU 0.3519 and nearly identical class counts (early ~1540, late ~12000, persistent
  ~6900). The main 1B run has fewer late (9673) because it trained 10x longer with a different loop sampling.
- Decoder matching (Hungarian): mean matched cosine 0.51 (s1~s2), 0.55 (seed~1B); only 7-8% of features match at
  cos > 0.9, 27-34% at cos > 0.7. So **individual features are not reproducible across seeds** — the usual SAE
  instability; roughly a third of the dictionary is shared as recognisable directions.
- **The early/late/persistent split is reproducible even though individual features are not.** Among features that
  *do* match across seeds (cos > 0.7), class agreement is 0.89 for early, 0.75 for late, 0.73-0.80 for persistent,
  far above the chance shares (0.05-0.06 / 0.30-0.37 / 0.21-0.23). A feature that is early in one SAE, if it has a
  counterpart in another SAE, is early there too ~89% of the time. So the population-level early/late conclusion
  (B1 and the shared-SAE analysis) is seed-robust; claims about specific feature ids are not.
