# Running the fine-tuning on Kaggle

## Why

The project was developed on a fanless MacBook Air M4. Measured, on one and the same task:

| condition | time |
|---|---|
| cold machine | **4m13s** |
| after ~30 min of sustained GPU load | **1h27m** |

A factor of 20, caused by thermal throttling of a passively cooled chassis. The problem is not only
that it is slow — it is that it is *unpredictable*, and that the slowdown grows with how much work
has already been done. Since every task vector in this study has to be produced under identical
conditions, running them on hardware whose speed drifts by 20x over a session is also a
methodological risk, not just an inconvenience.

All 48 fine-tunings are therefore produced on one machine, in one place, with one backend.

## One-time setup

1. Create a **public** GitHub repository (the report has to link to it anyway) and push this project.
2. Go to <https://www.kaggle.com/code> → **New Notebook**.
3. Right-hand panel → **Settings**:
   - Accelerator → **GPU T4 x2** (P100 is fine too)
   - Internet → **On** (needed for `pip`, `git clone`, and the CIFAR-100 download)
4. File → **Import Notebook** → upload `notebooks/kaggle_finetune.ipynb`.
5. In the first code cell, set `REPO_URL` to your repository.

## Running

Run the cells in order. Cells 2–4 are cheap sanity checks and should all pass before any GPU time is
spent; in particular the self-test must report **13 passed, 0 failed**.

Cell 5 prints the *measured* sustained throughput and the projected total. Read it before starting
the long run: if the projection does not fit in a session, reduce `epochs` in the config rather than
discovering the problem six hours in.

**The run is resumable.** Each task vector is written the moment its task finishes, and completed
tasks are skipped on restart. A session timeout costs only the task in flight — re-run the notebook
and it continues where it stopped.

## Getting the results back

`Save Version` → `Save & Run All` produces a versioned output you can download, or attach to a
follow-up notebook as a dataset. The archives are built by the last cell:

| file | contents | approx. size |
|---|---|---|
| `task_vectors.zip` | `checkpoints/` — every tau, fp16 | ~700 MB |
| `metrics.zip` | `results/` — per-task accuracies, JSON | < 1 MB |

Only `metrics.zip` is needed to start the analysis. Download `task_vectors.zip` when you want to
compute the weight-space predictors locally.

## Note on the local checkpoints

Any `checkpoints/` produced locally on MPS are **not** used. They are excluded by `.gitignore`, so
they never reach the Kaggle clone, and the run there starts from an empty directory. Mixing task
vectors trained on different backends would introduce exactly the kind of confound this project is
built to avoid.
