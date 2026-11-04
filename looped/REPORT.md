# Reproduction of arXiv 2604.11791 on Ouro-1.4B and Huginn-0125

Paper: Blayney et al., "A Mechanistic Analysis of Looped Reasoning Language Models" (2026).
Only the reproduced parts are listed. For each item: the paper's claim -> whether we reproduce it -> where the
results are -> what is wrong with / needs qualifying in the original claim.

Paths: figures `looped/results/{ouro,huginn,compare}/figs/` (compare figures directly in `looped/results/compare/`),
numbers `looped/results/{ouro,huginn,compare}/summary.json`, raw per-example data
`/data_fs/zhangyuxuan/looped/results/{ouro,huginn}/<run>/ex*.npz`. Below, `O:` = `looped/results/ouro/figs/`,
`H:` = `looped/results/huginn/figs/`, `C:` = `looped/results/compare/`.

## Setup

| | Ouro-1.4B | Huginn-0125 |
|---|---|---|
| Structure (prelude, block, coda) | (0, 24, 0), no input injection | (2, 4, 2), linear input injection |
| Norm placement | sublayer outputs normed; residual normed only at end of each loop | residual normed after every sublayer (norm_2, norm_4) |
| Train recurrence | fixed 4 | sampled, mean 32 (truncated backprop, 8 steps) |
| BOS | none (sink sits on first real token) | prepended |

- Data: 256 random GSM8k test questions (seed 0; question only, no template, no system prompt); 256 HellaSwag
  test contexts for the non-reasoning check; single prompts "Hello! I've been well. I hope that you're doing well."
  and "The square root of 16 is" for trajectories.
- bf16 weights, metrics in fp32. Ouro uses eager attention; Huginn attention probabilities are recomputed from
  the block's own q/k (verified: reproduces the attention output to 0.27% relative error; the last-loop readout equals
  the model's logits exactly). Huginn's random initial state is seeded per example; a second seed gives identical
  conclusions (`H:*_gsm8k_L32_seed1000.png`).
- "Residual after layer l" = output of recurrent-block layer l (Ouro: before end-of-loop norm; Huginn: after
  norm_4, with the pre-norm value also stored). Metrics averaged over tokens, heads, examples.
- Feedforward reference: Llama-3.2-1B (`looped/results/llama/llama_baseline.json`).
- Ouro bf16 vs fp32: all 8-loop attention metrics agree to <= 0.002 and the 128-loop dynamics are the same
  (`O:*_gsm8k_L128_fp32.png`), so nothing below is a precision artifact.

**Key confound (applies to items 3-6).** The paper evaluates all models at 8 and 128 recurrences. For Huginn,
128 = 4x its mean train recurrence; for Ouro, 128 = 32x its train recurrence of 4. Ouro's own readout CE is
best at the train depth and collapses with extrapolation (GSM8k, mean over 256 questions):

| loops | 1 | 2 | 3 | 4 | 8 | 16 | 24 | 32 | 48 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ouro CE | 3.74 | 2.38 | 2.00 | **1.96** | 1.99 | 2.54 | 4.18 | 6.58 | 10.76 | 12.08 | 12.36 |
| Huginn CE | 4.12 | 3.34 | 2.96 | 2.75 | 2.43 | 2.36 | 2.35 | **2.35** | 2.35 | 2.35 | 2.35 |

So comparisons at a fixed absolute number of recurrences compare Huginn near its training regime with Ouro far
outside it (Ouro at 128 loops is a broken model). The paper attributes Ouro's behaviour to architecture
(no input injection / norm placement); with the available models this cannot be separated from train depth.

---

## 1. Attention patterns of a layer are most similar to the same layer at other recurrences (Fig 2, Prop 4.2)

**Paper.** Frobenius distance between attention matrices over all depth pairs (8 loops) shows diagonal stripes:
each layer's attention is closest to the same layer in other loops. For Ouro this converges "after the first iteration".

**Reproduced: yes, both models.**

| | Ouro (8 loops) | Huginn (8 loops) | Huginn (32 loops) |
|---|---|---|---|
| Nearest depth (attention) is the same layer, loops >= 2 | 100% | 100% | 100% |
| ‖ΔA‖_F same layer, consecutive loops (loops >= 2) | 0.66 | 1.61 | 0.52 |
| ‖ΔA‖_F adjacent layers, same loop | 2.94 | 4.62 | 4.44 |
| ‖ΔA‖_F same layer, loop 1 -> 2 | 6.30 | 3.42 | 3.42 |

Results: `O:fig2_19_heatmaps_gsm8k_L8.png`, `H:fig2_19_heatmaps_gsm8k_L8.png`, `H:fig2_19_heatmaps_gsm8k_L32.png`;
HellaSwag `O:fig2_19_heatmaps_hellaswag_L4.png`, `H:fig2_19_heatmaps_hellaswag_L32.png`;
summary keys `*_nearest_attn_is_same_layer_frac`, `*_attnF_*`.

**Qualification.** For Ouro, "converges after the first iteration" is accurate in the sense that loop 1 is the odd one
out: its attention is further from loop 2 (6.30) than adjacent layers are from each other (2.94), and the loop-1 row
of the heatmap has no same-layer stripe. Loops 2-8 are nearly identical. Huginn converges more gradually
(3.42 -> 2.58 -> 2.01 -> ... -> 0.07 by loop 32).

## 2. Residual streams show the same cyclic structure (Fig 19)

**Paper.** Residual cosine similarity between all depth pairs shows the cyclic pattern; each layer reaches a different
fixed point.

**Reproduced: yes, but the raw-residual measure is weak evidence.**
Nearest depth by residual cosine is the same layer for 76% (Ouro, 8 loops) and 54% (Huginn, 8 loops) of depths.

Results: right panels of the `fig2_19_heatmaps_*` figures; centered cosine and CKA in `O:geometry_gsm8k_L8.png`,
`H:geometry_gsm8k_L8.png`; update-vs-residual numbers from `code/delta_geometry.py`
(`looped/results/{ouro,huginn}/delta_geometry_gsm8k_L8.npz`).

**Problem with the original measure.** Raw residual cosine mixes two things: (a) inheritance through the residual
stream (x_l = x_{l-1} + u_l, with ‖u‖/‖x‖ ≈ 0.4), which makes adjacent layers similar regardless of what they compute,
and (b) the per-layer computation. Separating them (loops >= 3, centered cosine):

| | Ouro residual | Ouro update u_l | Huginn residual | Huginn update u_l |
|---|---|---|---|---|
| adjacent layers | 0.92 | **-0.02** | 0.90 | **0.06** |
| same layer, consecutive loops | 0.94 | **0.91** | 0.91 | **0.89** |
| nearest is same layer | 82% | **100%** | 61% | **100%** |

The cyclic, per-layer structure is much cleaner in what each layer *writes* (100% same-layer, adjacent layers write
nearly orthogonal directions) than in the residual itself. Subtracting the mean direction changes little
(raw vs centered cosine within 0.01-0.1). Linear CKA is not suitable for this question: it is dominated by token
identity shared across all depths (Huginn CKA > 0.7 everywhere; Ouro loops >= 3 all > 0.8), so it shows almost no
per-layer structure.

## 3. Successive recurrences change little (Fig 3, 24, 25)

**Paper.** Norm / cosine / attention difference between the same layer at consecutive recurrences becomes small for all models.

**Reproduced: yes.** Mean over layers of cos(x_{l,t}, x_{l,t-1}):

| t | 2 | 3 | 4 | 8 | 16 | 32 | 128 |
|---|---|---|---|---|---|---|---|
| Ouro | 0.72 | 0.88 | 0.93 | 0.99 | 0.95 | 0.98 | 0.99 |
| Huginn | 0.63 | 0.77 | 0.85 | 0.96 | 0.997 | 1.000 | 1.000 |

Results: `O:fig3_4_22_25_dynamics_gsm8k_L8.png`, `O:fig3_4_22_25_dynamics_gsm8k_L128.png`,
`H:fig3_4_22_25_dynamics_gsm8k_L128.png` (top rows).

**Qualification.** For Ouro the successive difference is not monotone: it dips at t = 8 and rises again at
t = 16-24 (cosine 0.95; attention difference 0.26 -> 0.97), i.e. exactly where the model leaves its train regime (item 5).

## 4. Huginn reaches a fixed point; Ouro does not (Fig 4, 22, 23)

**Paper.** Distance of each layer's residual / attention to its value at the 128th recurrence ("approximate fixed point"):
Huginn converges quickly; Ouro does not, despite small successive differences. Attributed to input injection.

**Reproduced: yes, the measurements.**

| t | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| Ouro cos(x_t, x_128) | 0.11 | 0.12 | 0.13 | 0.17 | 0.31 | 0.60 |
| Huginn cos(x_t, x_128) | 0.37 | 0.74 | 0.92 | 0.99 | 0.999 | 1.000 |
| Ouro ‖A_t − A_128‖_F | 5.91 | 6.80 | 6.71 | 6.05 | 3.68 | 2.18 |
| Huginn ‖A_t − A_128‖_F | 3.51 | 1.87 | 0.99 | 0.34 | 0.10 | 0.07 |

Results: bottom rows of `O:fig3_4_22_25_dynamics_gsm8k_L128.png` and `H:fig3_4_22_25_dynamics_gsm8k_L128.png`;
single prompts `*_hello_L128.png`, `*_sqrt16_L128.png`.

**Problem with the original interpretation.** For Ouro the reference state (loop 128) is not a fixed point of the
trained computation but a degenerate regime: CE ≈ 12.4 there, and sinks have vanished (item 5). Low similarity to it
says Ouro left its working regime, not that its trained dynamics fail to converge. Within / near its train depth
Ouro is stable (successive cosine 0.93-0.99 for t = 4-8, CE flat 1.96-1.99). Since Huginn is only evaluated up to
4x its train depth and Ouro up to 32x, the data do not isolate the effect of input injection.

## 5. Ouro's stages keep changing over 128 recurrences, Huginn's are constant (Fig 11, 12, 47, 50)

**Paper.** Per-layer ColSum concentration over 128 recurrences: retrofitted/Huginn layers settle to constant values;
Ouro's layers "continually change throughout the recurrences tested" and deviate from train-time stages.

**Reproduced: Huginn yes; Ouro: the deviation yes, "continually change" no.**

| t | 4 | 8 | 16 | 24 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|
| Ouro sink rate (mean over layers) | 0.95 | 0.94 | 0.75 | 0.25 | 0.07 | 0.04 | 0.03 |
| Ouro ColSum (mean over layers) | 0.58 | 0.56 | 0.45 | 0.25 | 0.20 | 0.27 | 0.32 |
| Ouro mean |ColSum − loop-4 profile| | 0 | 0.017 | 0.13 | – | 0.38 | 0.31 | 0.26 |
| Huginn mean |ColSum − loop-32 profile| | 0.005 | 0.003 | 0.000 | – | 0 | 0.000 | 0.000 |

Results: `O:fig11_47_stability_by_layer_gsm8k_L128.png`, `O:fig12_50_stability_by_depth_gsm8k_L128.png`,
`H:fig11_47_stability_by_layer_gsm8k_L128.png`, `H:fig12_50_stability_by_depth_gsm8k_L128.png`
(red line / dashed curve = each model's own train depth); summary keys `*_colsum_dev_from_train_depth_*`.

**Problem with the original claim.** Ouro does not drift continually. It has three regimes: (i) stable around the
train depth (loops 2-12, stages essentially fixed); (ii) a transition at roughly 4-8x train depth (loops ~15-30)
in which attention sinks disappear (sink rate 0.95 -> 0.03) and ColSum collapses; (iii) a new, slowly settling regime
after ~40 loops whose stages differ from the trained ones and where the model is broken (CE ≈ 12). The relevant
comparison with Huginn would be at matched multiples of train depth, which the paper does not make.

## 6. Latent trajectories (Fig 26, 28; Huginn cyclic fixed point, Fig 1/5)

**Paper.** Huginn's last-token trajectory settles onto a fixed cycle; Ouro reaches an approximately constant trajectory
with larger deviations; on "The square root of 16 is" Ouro is stable at recurrences 8-16 and then becomes unstable.

**Reproduced: yes.**
- Huginn: per-loop trajectories converge and from ~loop 33 on overlap exactly (`H:fig26_28_pca_sqrt16_L128.png`,
  `H:fig26_28_pca_hello_L32.png`).
- Ouro "sqrt 16": loops 1-16 overlap, 17-24 start to leave, 25-32 fully depart (`O:fig26_28_pca_sqrt16_L32.png`).
  "Hello" prompt: stable at 9-24, departs at 25-32 (`O:fig26_28_pca_hello_L32.png`).

**Qualification.** Ouro's departure point (16-24 loops) coincides with the regime change of item 5, i.e. with leaving
the trained regime, not with an intrinsic instability of the trained dynamics.

## 7. Each loop of Ouro repeats feedforward-like stages of inference; Huginn shows no clear stages (Fig 8, 35, 36)

**Paper.** Within each recurrent block, ColSum concentration closely mirrors a feedforward model (Ouro ~ Llama),
repeated every loop. Huginn does not show clear stages. Ouro's residual entropy changes across recurrences.

**Reproduced: yes for Huginn; for Ouro the stages exist but the match to Llama is moderate.**

| | Ouro (per loop 1/2/3/4) | Huginn |
|---|---|---|
| Within-loop range of ColSum (Llama: 0.37) | 0.34 | 0.18 |
| Within-loop range of mixing (Llama: 0.33) | 0.38 | 0.10 |
| Within-loop range of residual entropy (Llama: 0.27) | 0.42 | 0.02 |
| Pearson with Llama, ColSum | 0.53 / 0.34 / 0.30 / 0.31 | n/a (4 layers) |
| Pearson with Llama, mixing | 0.69 / 0.57 / 0.45 / 0.46 | n/a |

Ouro residual entropy at layer 0: 0.11 / 0.68 / 0.76 / 0.78 for loops 1-4 (changes across recurrences, as stated).
Huginn residual entropy is flat at 0.85-0.88 in every layer and loop.

Results: `C:fig8_35_stages_vs_feedforward.png`, `O:fig8_35_stages_gsm8k_L4.png`, `O:fig9_34_realized_depth_gsm8k_L8.png`,
`H:fig8_35_stages_gsm8k_L32.png`, `H:fig9_34_realized_depth_gsm8k_L32.png`; `looped/results/compare/summary.json`.

**Qualification.** "Closely mirrors" overstates it for Ouro: the qualitative shape (low-high-low mixing, ColSum rising
towards the end of the block) is shared, the correlation is 0.3-0.7. Part of the mismatch is the sink token:
Llama sinks on BOS, Ouro (no BOS) on the first real token, and Ouro has no sinks at all in loop 1 on GSM8k (item 9).
Huginn has no feedforward base model; Llama-3.2-1B is used as a generic reference.

## 8. Huginn cannot develop residual-norm growth because it normalises the residual (Fig 9)

**Paper.** Huginn normalises the residual stream after each sublayer and therefore lacks the residual-norm growth that
produces massive activations, sinks and stages in feedforward models; Ouro normalises only at the end of each loop.

**Reproduced: yes.**

| | min ‖x‖ | max ‖x‖ | max/min |
|---|---|---|---|
| Llama-3.2-1B (one pass) | 1.6 | 27.0 | 16.7 |
| Ouro (within a loop) | 2.0 | 32.5 | 16.4 |
| Huginn after norm_4 | 70.2 | 76.4 | 1.09 |
| Huginn before norm_4 | 66.1 | 78.4 | 1.19 |

(mean norm over non-initial tokens.) Position-0 / median norm ratio: Ouro 10-16 inside loops >= 2; Huginn 1.0-2.4.
Results: `C:fig9_residual_norms.png`.

**Problem with the original causal story.** Huginn has no massive activation on its sink token, yet it still has
attention sinks: the most-attended key is position 0 (BOS) in 100% of layers and loops, sink rate 0.57-0.83
(`H:sinks_gsm8k_L8.png`). So "no norm growth -> no massive activations" holds, but sinks in Huginn do not depend on
massive activations; the lack of stages cannot be explained by a lack of sinks.

## 9. Additional observation: sinks in Ouro appear only from loop 2 on (GSM8k)

Not a paper claim, but it qualifies items 1 and 7. On GSM8k, Ouro's loop 1 has essentially no attention sink
(sink rate 0.03; the most-attended key is position 0 in only 1.6% of layers), while from loop 2 on position 0 is the
most-attended key in 96-99% of layers with sink rate ≈ 0.95. This coincides with the position-0 norm ratio jumping
from 3-5 (loop 1) to ≈ 16 at the start of loop 2. On HellaSwag the sink already forms in the second half of loop 1
(sink rate 0.63 in layers 12-23 of loop 1). Results: `O:sinks_gsm8k_L8.png`, `O:sinks_hellaswag_L4.png`.

## Implementation notes

- Code: `code/models/{ouro,huginn,llama}.py` (adapters), `code/collect.py`, `code/analyze.py --model {ouro,huginn}`,
  `code/compare.py`, `code/geometry.py`, `code/delta_geometry.py`, `code/sanity.py`, `code/launch_{ouro,huginn}.sh`.
- Metric definitions: ColSum concentration as in the paper; sink rate with tau = 0.3 on position 0; mixing score
  normalised by log(#keys); residual entropy = alpha = 1 matrix entropy of the token Gram matrix, normalised by log T
  (the paper does not state alpha).
