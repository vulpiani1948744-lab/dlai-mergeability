# When Do Models Merge?

**Predicting merging success from weight- and representation-space signals.**

Project for *Deep Learning & Applied AI* (a.y. 2025/2026), Sapienza University of Rome.

Repository: <https://github.com/vulpiani1948744-lab/dlai-mergeability>

> Model merging combines models fine-tuned from a shared checkpoint into one multi-task model,
> without retraining. It works often, and fails silently. We ask whether failure is predictable
> **before** merging, from cheap signals computed on the individual checkpoints.

See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the full experimental design.

## Setup

```bash
uv sync
```

## Reproducing the experiments

```bash
uv run scripts/run_finetune.py --config configs/benchmark_a.yaml   # fine-tune task-specific models
uv run scripts/run_merge.py    --config configs/benchmark_a.yaml   # merge + evaluate
uv run scripts/run_predictors.py --config configs/benchmark_a.yaml # pre-merge signals
uv run scripts/run_analysis.py                                     # correlations, regression, figures
```

## Repository layout

```
src/mergeability/
  data.py            dataset construction (CIFAR-100 splits, small vision datasets)
  models.py          backbone loading, task-vector extraction
  finetune.py        fine-tuning loop
  merging/           averaging, task arithmetic, TIES, DARE, TSV
  similarity/        CKA, relative representations, subspace angles, sign conflict
  analysis.py        correlation + regression analysis
  plots.py           figure generation
scripts/             entry points
configs/             experiment configurations
results/             metrics (csv) and figures
report/              LaTeX report on the official dlaiml2026 template
docs/                project plan, notes
```

## Authors

- Chiara Vulpiani — 1948746
- Rachele Vulpiani — 1948744

Every stage of this project — design, implementation, experiments, analysis and report — was
carried out jointly by both authors.

## Use of AI tools

See [`AI_STATEMENT.md`](AI_STATEMENT.md).
