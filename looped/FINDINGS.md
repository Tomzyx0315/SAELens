# Findings: looped language models through mechanistic analysis and SAEs

Summary of the project. Models: Ouro-1.4B and Huginn-0125 (looped / recurrent-depth LMs), with Llama-3.2-1B as a
feedforward reference. Two parts: (1) reproduction of arXiv 2604.11791 on both models (`REPORT.md`); (2) a
sparse-autoencoder study of Huginn's cross-loop representations (`SAE.md`). Data in `DATA.md`.

## Headline result

Huginn's recurrence has two regimes, and the SAE evidence makes the boundary concrete:

- **Loops 1-4 build the representation in an evolving, lower-dimensional space; loops ~8-32 share one converged
  space and mostly consolidate.** Per-loop SAEs are interchangeable among loops 8/16/32 (cross-loop reconstruction
  FVU within 0.005 of each loop's own SAE) but fail across loop 1 (~0.42 excess FVU). So the loop representations
  are **not** one shared space with features fading in and out; there is a genuine change of representation space
  over the first few loops, then a fixed point.

This is supported independently by geometry (loop 1 participation ratio ~287 vs ~1200 later), attention/stage
dynamics (`REPORT.md`), the per-loop feature turnover, and causal splicing.

## Part 1 — Reproduction (`REPORT.md`)

Reproduced on Ouro and Huginn, with corrections to the paper where our data disagree:

1. Attention of a layer is most similar to the same layer at other recurrences (cyclic structure): reproduced
   (100% same-layer nearest neighbour at loops >= 2 for both models).
2. Huginn reaches a cyclic fixed point; Ouro does not — reproduced, but the comparison is confounded by train
   depth (Huginn trained at mean 32 loops, Ouro at 4; "128 loops" is 4x vs 32x their training). Ouro at 128 loops
   is a broken model (CE ~12), so its "non-convergence" is extrapolation, not an architecture verdict.
3. Each loop repeats feedforward-like stages of inference (Ouro ~ Llama): partially — the shape matches, the
   correlation is only 0.3-0.7; "closely mirrors" overstates it.
4. Huginn normalises the residual every sublayer and so has no massive activations: reproduced — but Huginn still
   has attention sinks without massive activations, so the paper's "no norm growth -> no sinks" causal chain does
   not hold for Huginn.
5. Our own additions: centered-cosine / CKA / per-layer-update geometry (the clean cyclic structure is in what each
   layer *writes*, not in the residual); Ouro sinks form only from loop 2 on (GSM8k).

## Part 2 — Cross-loop SAE on Huginn (`SAE.md`)

Site: the loop state (output of the last core-block layer, d 5280). Main SAE: BatchTopK, d_sae 32768, k 128,
1e9 activation vectors, held-out FVU 0.32 (inference JumpReLU faithful, 64 dead features).

### Feature dynamics across loops
- The recurrence mostly **adds** features: 1741 early (loop-1-only), 11940 late-emerging (loop-32-only), 6029
  persistent. Per-loop feature usage is fully determined by computation, not by the random init state
  (seed-vs-seed frequency correlation 1.00).
- Quantitative descriptors (A4): the defensible contrast is **early = few, frequent, domain-general; late = many,
  rare, domain-specific** (domain entropy 0.81 vs 0.62, p=2e-143). The earlier "syntax first, content later" claim
  did **not** survive the statistics (token entropy and punctuation share are indistinguishable) and was retracted.
- Feature flow with baselines (A2): feature turnover tracks the size of the state change almost 1:1, so "turnover
  decays over loops" is just state convergence; but for the same state distance, real loop-to-loop moves turn over
  ~1.3-1.5x more feature mass than isotropic noise, i.e. the trajectory moves along SAE-resolved directions.
- Feature lineage (A3): when features switch off, the features that switch on point in **near-orthogonal**
  directions (decoder cosine ~0, token-level cos(v_off, v_on) 0.04-0.13) — the representation genuinely changes
  content across loops, it is not the same concept re-encoded by another latent.

### Faithfulness (how much to trust the above)
- Per-depth splicing (A1): replacing the loop-l state with its SAE reconstruction and reading out at depth l
  recovers 0.94-0.95 of the mean-ablation loss at every depth, **worst at loop 1 (0.86)**. delta CE ~0.28-0.53 nat.
  This is the faithfulness ceiling of all feature-level claims.
- Single-edit splice into the full forward is uninformative at intermediate loops (the recurrence overwrites any
  intermediate state; even mean-ablation has no effect there). Only the read-out loop (32) matters for the output.

### Is the space the same across loops? (B — the direct test)
- B1 transfer matrix: loops 8/16/32 are one space (interchangeable SAEs); loop 1 is a distinct space (~0.42 excess
  FVU to/from it); loops 2-4 are the transition. Decoder matching agrees (near-identical features only among late
  loops).
- B2 crosscoder: forcing shared latents, no latent is purely early or late — the early/late distinction lives in
  the per-loop encoder/decoder weights, not in a subset of latents. (This run had mis-scaled aux loss and ~2000
  dead latents; distribution still informative, clean re-run on backlog.)
- B3 seed stability: individual features are not reproducible across init seeds (7-8% match at cos>0.9), the usual
  SAE instability — but the **early/late/persistent split is seed-robust** (89% class agreement among matched early
  features vs 5% chance). Population-level conclusions hold; specific feature ids do not.

## What is solid vs tentative

Solid: the two-regime picture and the loop-8+ shared space (B1, multiple independent measurements); the population
early/late split and its seed-robustness; the reproduction claims with their corrections.

Tentative / limited: interpretations of individual features (need auto-interp with an LLM we have not set up);
the crosscoder absolute dead count; everything is one SAE site (loop state), so flow *across the 4 core layers*
within a loop is untested; no causal steering yet; Huginn only (train-depth confound not isolated).

## Backlog (see the TODO list)
- C: causal interventions — persistent steering/ablation of candidate features across loops; test whether early
  features cause late features.
- D: per-core-layer SAEs (the "different layers" question); cross-model / train-depth controls.
- crosscoder clean re-run; auto-interp.

## Where things are
- Docs: `REPORT.md`, `SAE.md`, `DATA.md`, this file. Code: `looped/code/` (`paths.py` for layout).
- Results: `looped/results/{ouro,huginn,compare,sae,llama}/`; raw data and SAE checkpoints on
  `/data_fs/zhangyuxuan/looped/`.
