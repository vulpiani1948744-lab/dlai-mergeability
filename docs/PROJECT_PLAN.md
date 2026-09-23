# Project Plan — Is Mergeability Algorithm-Relative?

**Course:** Deep Learning & Applied AI, a.y. 2025/2026 (Prof. E. Rodolà, Sapienza)
**Format:** BYOP, 2-student team → 3-page report
**Target session:** extraordinary session, autumn 2026 → **compressed 3-week plan**
**Guideline areas:** *model merging*, *representation learning* (secondary: *training dynamics*, *multitask learning*)

---

## 0. The one-paragraph version

Model merging fuses several models fine-tuned from a shared checkpoint into one multi-task model,
without retraining. It works often and fails silently. We fine-tune the same backbone on task
families whose **inter-task similarity we control on purpose**, merge them with four algorithms that
fail for structurally different reasons, and ask which cheap pre-merge signal predicts the damage.
Our claim is that there is no single such signal: **mergeability is a property of the (model pair,
algorithm) couple, not of the model pair alone.**

## 1. The question

> Given two models fine-tuned from the same checkpoint, does the signal that predicts merging damage
> depend on *which merging algorithm* you intend to use?

**Why no outcome wastes the work.** If the best predictor changes with the algorithm, mergeability is
algorithm-relative and the practical recipe is "pick the diagnostic that matches your merger" — a
result. If one signal dominates everywhere, mergeability is an intrinsic property of the model pair
and a single cheap diagnostic suffices — a cleaner, more useful result. If nothing predicts anything
above the trivial baselines, that is a negative result about the whole diagnostic programme, and the
controlled similarity axis (§3.1) proves the signal was there to be found.

## 2. The controlled comparison

**The one variable we change on purpose:** inter-task similarity.
**Everything held fixed**, and tabulated in the report:

| | value |
|---|---|
| Backbone | identical, same pretrained checkpoint $\theta_0$ |
| Architecture of every merged model | **bit-identical** — we merge encoders only |
| Classification heads | frozen before fine-tuning (§3.2), shared across all encoders |
| Optimizer / LR / schedule / epochs / batch | identical across all tasks |
| Train budget | 10k train, 2k test images per task |
| Seeds | 0, 1 — the same seeds for every condition |

Note what this buys us over a typical comparison: because only the encoder is merged and the heads
are frozen, **there is no architectural asymmetry at all** between the conditions. No parameter-count
confound to declare.

## 3. Setup

Shared pretrained backbone $\theta_0$; fine-tune on $T$ tasks → $\theta_1 \dots \theta_T$;
task vectors $\tau_i = \theta_i - \theta_0$.

### 3.1 Benchmark A — the controlled similarity axis (fit set)

CIFAR-100 partitioned into 5 disjoint 20-class tasks, in two regimes:

- *semantic* splits — each task is the union of 4 related superclasses, so the tasks demand
  genuinely different features (fur and muzzles vs. wheels and tarmac) → **low inter-task
  similarity**, task vectors point in different directions, merging should be **hard**.
- *random* splits — the 100 fine classes are shuffled and cut arbitrarily, so every task is a
  microcosm of the whole dataset and they all demand nearly the same features → **high inter-task
  similarity**, task vectors point in nearly the same direction, merging should be **easy**.

This is the dial. Same images, same class count, same training budget, same everything — and
mergeability high or low on command. Without it we would only be observing correlations; with it we
have a designed experiment. No extra downloads.

The two regimes also double as a **sanity check on the whole diagnostic programme**: any predictor
worth reporting must at minimum separate these two regimes, because we know by construction that
they differ.

### 3.2 Classification heads: linear-probe-then-freeze

If each task's head were trained with its own encoder, the head would be matched to *that* encoder
and not to the merged one; the measured drop would confound head mismatch with merging failure. So:

1. extract features from the **frozen** pretrained backbone $\theta_0$;
2. fit a linear probe per task (logistic regression on cached features — seconds);
3. **freeze** the head, fine-tune only the encoder.

Heads become shared across the whole $\tau$ family, which is what makes task arithmetic well-posed.
This plays the role CLIP's frozen zero-shot heads play in the task-arithmetic literature, without
paying for CLIP.

### 3.3 Benchmark B — held-out heterogeneity (transfer set)

6 genuinely different datasets: CIFAR-10, SVHN, GTSRB, EuroSAT, DTD, FashionMNIST.
The predictor is **fitted on A and evaluated here**. A real generalization claim, not an in-sample fit.

### 3.4 Backbones

- **ViT-Tiny** (5.7M, ImageNet-pretrained) @224 — primary. A transformer, so layer-wise CKA is
  meaningful and it connects to the course's attention material.
- **ResNet-18** @64 — the **cross-architecture control** (§6). Cheap.

## 4. Merging algorithms — chosen because they fail differently

| # | Method | Mechanism it relies on | Predicted failure mode |
|---|---|---|---|
| 1 | Weight averaging | none — pure mean | representation drift from $\theta_0$ |
| 2 | Task arithmetic ($\lambda$ sweep) | global rescaling | task-vector interference (cosine) |
| 3 | TIES-Merging | trim, elect sign, disjoint merge | sign conflict |
| 4 | DARE + TIES | stochastic sparsification + rescale | task-vector density / effective rank |

This is the heart of the design: four *distinct mechanisms*, each with a **pre-registered prediction**
about which signal should matter for it. The predictions are written down before the experiments run.

**Metric.** Normalized accuracy $\frac{1}{T}\sum_i \mathrm{acc}_{\text{merged}}(i)/\mathrm{acc}_{\text{finetuned}}(i)$ —
standard in the merging literature. Reported with min/max over seeds.

## 5. Baselines — the part we were missing

No predictor counts as informative until it beats these. This is what makes a correlation mean
something.

|      | baseline                                          | what it rules out                                                       |
| ---- | ------------------------------------------------- | ----------------------------------------------------------------------- |
| `b0` | predict the global mean normalized accuracy       | that any structure exists at all                                        |
| `b1` | predict from the **number** of merged tasks alone | that we are just measuring "more tasks is worse"                        |
| `b2` | predict from $\lVert \tau \rVert$ alone           | that we are just measuring "these models moved further from $\theta_0$" |

`b2` is the dangerous one, and the analogue of "paint everything the average colour": a large task
vector trivially predicts a large merge error. Any of our signals that does not beat `b2` is
measuring distance travelled with extra steps, and gets reported as such.

## 6. The control that kills the alternative explanation

Our main claim is that the predictor ranking **changes across merging algorithms**. The obvious
objection is that the ranking is unstable noise. So we check that the same ranking is **stable**
along two axes where it should not change:

- across **seeds** (0, 1) — same task family, different initialization of fine-tuning
- across **backbones** (ViT-Tiny vs ResNet-18) — different architecture, same task family

If the ranking is stable across seeds and architectures but moves across algorithms, the
algorithm-relative claim survives. If it is unstable everywhere, we report that we cannot support it.

## 7. Pre-merge predictors

Computed from checkpoints alone, plus a fixed unlabeled probe set (~2000 images) shared across tasks.

**Main text (5)**
- `p1` mean pairwise cosine similarity between task vectors
- `p2` magnitude-weighted sign-conflict rate
- `p3` principal angles between top-$k$ singular subspaces of layer-wise reshaped $\tau_i$
- `p4` layer-wise linear CKA between $\theta_i$ and $\theta_j$ (Kornblith et al., 2019)
- `p5` representation drift: CKA between $\theta_i$ and $\theta_0$

**Appendix**
- effective rank / spectral entropy of $\tau_i$; relative-representation agreement
  (Moschella et al., ICLR 2023). ($\lVert\tau\rVert$ is promoted to baseline `b2`.)

## 8. Analysis

1. Spearman $\rho$ of each predictor vs. normalized accuracy, **per algorithm**, against `b0`–`b2`.
2. Ridge regression on the 5 predictors; fit on A, report $R^2$ and rank correlation on B.
3. Predictor-importance ranking per algorithm + the stability check of §6.
4. Layer-wise ablation: which layers' CKA carries the signal?

## 9. Figures (~4 fit in 3 pages)

- **F1** pipeline schematic: $\theta_0 \to \tau_i \to$ merge $\to$ predictors, with the similarity dial
- **F2** the controlled axis: normalized accuracy, semantic vs random splits, per algorithm
- **F3** predictor-importance heatmap (predictor × algorithm) + the seed/backbone stability check
- **F4** predicted vs. actual on held-out Benchmark B, with the `b2` baseline drawn in

## 10. Report discipline (adopted wholesale)

- **No number is typed by hand.** `scripts/report_numbers.py` reads the run logs and emits
  `report/numbers.tex`, whose first line is `% Generated by scripts/report_numbers.py. Do not edit.`
  Every figure in the prose is a macro.
- Every quoted number carries min/max over seeds.
- Confounds are declared in the Method section, not waited for.
- A "prediction that failed" paragraph goes **in** Results if we get one, not in a footnote.

## 11. Where this sits in the course

| What we use | Where the course teaches it |
|---|---|
| ViT / self-attention | 20 May, `14-trans.pdf`, Lab 11 |
| CNNs (ResNet control) | 31 Mar, `08-cnn.pdf`, Lab 06 |
| SVD / PCA of task vectors | 5 May, `12-pcavae.pdf` |
| Adam, LR schedules | 17 Mar, `06-sgd.pdf`, Lab 04 |
| `nn.Module`, autograd, state dicts | Lab 05 |
| Regularization, weight decay | 29 Apr, `09-regular.pdf`, Lab 07b |
| Low-rank structure of weight deltas | 22 Apr, quantization/pruning seminar, Labs 08a/08b |
| Representation similarity | representation-learning thread; relative representations (Moschella et al.) |

## 12. How the work was carried out

This is a two-student project and **every stage was carried out jointly**: the experimental design,
the implementation, the runs, the analysis and the report were developed together rather than split
into separate ownerships. Where the report needs to state this, it states exactly this.

## 13. Timeline (3 weeks, every milestone submittable)

| Week | Milestone | State at the end |
|---|---|---|
| 1 | Data, LP-then-freeze, Benchmark A fine-tuned (ViT-Tiny, 2 seeds, both regimes) | task vectors + per-task accuracies on disk |
| 2 | 4 merging algorithms + predictor library + baselines; A merged and analysed | F2/F3 drafted — **RQ already answered** |
| 3 | ResNet-18 control, Benchmark B transfer, figures, report, AI statement | submission-ready |

## 14. Risks

| Risk | Mitigation |
|---|---|
| MPS slower than estimated | 1 seed; 128×128; Colab T4 fallback |
| Predictor ranking is unstable | That *is* the §6 result, reported as a negative |
| Benchmark B downloads (~1 GB) slow | Start them in week 1, in the background |
| Week 3 slips | Project is already complete after week 2; B and the ResNet control are additive |

## 15. Deliverables checklist

- [ ] Public GitHub repository, reproducible (`uv sync` + `scripts/`)
- [ ] 3-page report on the official `dlaiml2026` template
- [ ] AI-use statement (before references, no length limit)
- [ ] Email to `{rodola,strano,solombrino}@di.uniroma1.it`, subject `[DLAI 2026] Project delivery`
- [ ] Both students registered on Infostud for the chosen session
- [ ] Open the repo link in a private window before sending
