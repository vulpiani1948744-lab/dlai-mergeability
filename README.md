# Is Mergeability Algorithm-Relative?

**Predicting model merging success from weight- and representation-space signals.**

Project for *Deep Learning & Applied AI* (a.y. 2025/2026), Sapienza University of Rome.

Model merging combines models fine-tuned from a shared checkpoint into one multi-task model, without
retraining. It works often, and fails silently. We ask whether the pre-merge signal that best
predicts merging damage depends on the merging algorithm. On CIFAR-100 task families whose
inter-task similarity is set by a five-level dial, with ViT-Tiny (and a ResNet-18 control), four
algorithms (weight averaging, task arithmetic, TIES, DARE) and five signals (cosine, sign conflict,
subspace overlap, CKA, drift), the answer is no: the same similarity signals predict the damage for
every algorithm, and CKA is the only signal consistent across both architectures. Details in the
report (`report/`).

## Repository layout

```
src/mergeability/
  data.py              CIFAR-100 task families and the similarity dial (semantic -> random)
  models.py            backbones, task vectors, frozen BatchNorm
  probe.py             linear-probe-then-freeze classification heads
  finetune.py          encoder-only fine-tuning
  merging/methods.py   weight averaging, task arithmetic, TIES, DARE
  similarity/          cosine, sign conflict, subspace overlap (weight space); CKA (representations)
  experiment.py        merge every subset of tasks with every algorithm, record outcome + signals
  analysis.py          per-algorithm correlations, baselines, stability, ridge regression
scripts/
  selftest.py          correctness checks (run first)
  check_resolution.py  linear-probe accuracy vs input resolution
  run_finetune.py      fine-tune one task vector per task, regime and seed
  verify_task_vectors.py  integrity check on task-vector norms
  run_merge.py         the merge experiment
  run_analysis.py      statistics -> results/analysis/
  report_numbers.py    every number quoted in the report -> report/numbers.tex
  make_figures.py      the report's figures -> report/fig_*.pdf
configs/               experiment configurations (benchmark_a = ViT-Tiny, benchmark_a_resnet = ResNet-18)
notebooks/             Kaggle notebook used for the GPU runs
results/raw/           merge results (CSV), fine-tuning log, resolution check
results/analysis/      analysis tables and figure previews
report/                LaTeX source of the report (official dlaiml2026 template)
docs/                  PROJECT_PLAN.md (written before the experiments), KAGGLE.md, evidence/
```

## How the results were produced

1. **Fine-tuning and merges** (70 task vectors; 2,310 ViT and 924 ResNet merges) ran on a Kaggle
   T4 GPU with `notebooks/kaggle_finetune.ipynb` (see `docs/KAGGLE.md`).
2. **DARE scaling sweep** (`run_merge.py --grid dare_sweep`, 630 ViT merges) was added after the
   first run and computed locally on Apple MPS from the same task vectors; reference accuracies and
   signals matched the GPU run. `results/raw/merge_vit_full.csv` is the union of the two
   (`merge_vit.csv` + `merge_vit_dare_sweep.csv`). The ResNet-18 control (`merge_resnet.csv`) has
   DARE at scaling 1.0 only.
3. **Fine-tuning log.** The per-task JSON files were lost with the Kaggle session;
   `results/raw/finetune_from_logs.csv` was reconstructed from the notebook output and checked
   against the task vectors.

Task vectors (~1 GB) and the dataset are not stored in the repository.

## Reproducing the analysis and the report (no GPU needed)

```bash
uv sync
uv run scripts/selftest.py         # dataset checks are skipped until CIFAR-100 is in data/
uv run scripts/run_analysis.py     # results/raw -> results/analysis
uv run scripts/report_numbers.py   # -> report/numbers.tex (needs CIFAR-100 in data/)
uv run scripts/make_figures.py     # -> report/fig_dial.pdf, report/fig_signals.pdf
```

## Reproducing the experiments (GPU)

```bash
uv run scripts/run_finetune.py --config configs/benchmark_a.yaml --device cuda
uv run scripts/verify_task_vectors.py checkpoints/vit_tiny_patch16_224
uv run scripts/run_merge.py --config configs/benchmark_a.yaml --device cuda --out results/raw/merge_vit_full.csv
```

The same commands with `configs/benchmark_a_resnet.yaml` run the ResNet-18 control. `run_merge.py`
saves its CSV after every (regime, seed) group, so an interrupted run keeps what it finished.

## Authors

- Chiara Vulpiani — 1948746
- Rachele Vulpiani — 1948744

Every stage of this project — design, implementation, experiments, analysis and report — was
carried out jointly by both authors.

## Use of AI tools

See the *Statement on the use of AI* in the report (`report/main.pdf`, Section 6).
