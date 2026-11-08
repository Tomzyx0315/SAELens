# Looped language models: mechanistic analysis + cross-loop SAEs

Research question: are the representations of a looped (recurrent-depth) language model consistent across loops
and layers? Studied on Ouro-1.4B and Huginn-0125.

**Start here: `FINDINGS.md`** — the project summary. Headline: Huginn's recurrence has two regimes; loops 1-4 build
the representation in an evolving, lower-dimensional space and loops ~8-32 share one converged space (per-loop SAEs
are interchangeable among late loops but not with loop 1).

| document | content |
|---|---|
| `FINDINGS.md` | Project summary: headline result, what is solid vs tentative, backlog |
| `REPORT.md` | Reproduction of arXiv 2604.11791 ("A Mechanistic Analysis of Looped Reasoning Language Models") on Ouro and Huginn: each paper claim, whether it reproduces, where the results are, what is wrong with it. Includes our own geometry analyses (centered cosine / CKA / per-layer updates) and the train-depth confound |
| `SAE.md` | Cross-loop SAE on Huginn's loop state: pipeline, config sweep, feature dynamics across loops, faithfulness checks (A1-A4), and the cross-loop space tests (B: transfer matrix, crosscoder, seed stability) |
| `DATA.md` | Provenance of every model weight and dataset (training, held-out, paper-reproduction, per-domain eval corpora) and download commands |

Layout:

```
code/            all code (paths in code/paths.py; model adapters in code/models/)
results/ouro/    paper-reproduction figures (figs/) and summaries for Ouro
results/huginn/  same for Huginn, plus the SAE feasibility check
results/llama/   feedforward baseline
results/compare/ cross-model figures
results/sae/     SAE run / sweep evaluation results (json)
results/logs/    job logs (*.log, not committed)
```

Large artifacts (weights, datasets, raw per-example data, SAE checkpoints, eval cache) are on
`/data_fs/zhangyuxuan/looped/` (see its README.md).
Environment: conda env `diffuser` (`/data/zhangyuxuan/miniconda3/envs/diffuser`), torch 2.12.1, transformers 4.56.2,
transformer-lens 2.17, SAELens from this repo (editable install).
