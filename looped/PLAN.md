# Reproducing arXiv 2604.11791 on Ouro-1.4B

Paper: "A Mechanistic Analysis of Looped Reasoning Language Models" (Blayney et al., 2026).
Goal: reproduce every Ouro-1.4B result, then use it as the foundation for SAE-based
cross-loop / cross-layer representation consistency studies.

## Ouro facts relevant to the measurements
- (0, 24, 0) structure, no input injection, trained with 4 loops (total_ut_steps=4).
- Block: X^ = X + n(Attn(n(X))); X' = X^ + n(MLP(n(X^))). The residual stream itself is only
  normalised at the end of each loop by `model.norm`, and the normed state is what feeds the next loop.
- Changing the number of loops = set `config.total_ut_steps`.
- No BOS is prepended (matches the paper's Table 2).

## Settings choices (not specified by the paper -> our choice, recorded here)
| Item | Choice | Reason |
|---|---|---|
| Data | GSM8k test, 256 random examples, seed 0, question text only, no chat template, no system prompt | paper: "without a system prompt" |
| Attention | eager, computed in fp32 from the hooked q/k (we recompute softmax(QK^T/sqrt d + mask) ourselves) | need explicit attention matrices |
| dtype | bf16 weights, fp32 metrics | model is bf16 native |
| Residual "after layer l" | output of decoder layer l (pre end-of-loop norm). Also log the post-`model.norm` state per loop | paper ambiguous; report both where it matters |
| Averaging | mean over tokens (valid positions) and examples; attention metrics also over heads | paper wording |
| Sink token | position 0 (first real token, no BOS) | Ouro has no BOS |
| sink-rate tau | 0.3 | App E.2 |

## Experiments (raw npz under /data_fs/zhangyuxuan/looped/results/<exp>; figures/summary under looped/results/)
1. exp1_cyclic (8 loops): Fig 2 attention Frobenius heatmap over all 192x192 depth pairs; Fig 19 residual cosine heatmap;
   Fig 3/24/25 successive differences (norm, cosine, attention).
2. exp2_fixedpoint (128 loops): Fig 4/22/23 distance to the "approximate fixed point" (128th loop) per layer;
   Fig 11/12/47/50 ColSum and other stage metrics per layer over 128 loops.
3. exp3_stages (4 and 8 loops): Fig 8/34/35: ColSum concentration, sink rate, mixing score, residual entropy per loop
   vs % depth; Fig 9 residual norm over depth.
4. exp4_pca (32 loops): Fig 26 test sentence "Hello! I've been well. I hope that you're doing well.", Fig 28 "The square root of 16 is".
5. exp5_hellaswag: Fig 42, exp3 on 256 HellaSwag test examples.
6. exp6_llama_baseline: Llama-3.2-1B feedforward stages (red dashed line in Fig 8/12), if download allowed.
7. Sanity: GSM8k/text perplexity vs number of loops 1..8 (paper cites Ouro Tab. 10: extrapolation hurts).

## Memory budget
128 loops x 24 layers = 3072 attention calls; attention matrices are never stored. Per call we reduce
online to: ColSum conc., sink rate, mixing score (per head -> mean), plus for pairwise Frobenius heatmaps
(8 loops only, 192 depths) we keep per-example attention (192 x 16 x T x T fp16) temporarily and reduce
per example. Residuals: store per example (depth, T, 2048) fp32 only as needed, reduce online.
