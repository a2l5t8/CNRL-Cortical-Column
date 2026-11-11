# Validation and provenance — 9 November 2026

| Protocol | Previous | New | Meaning |
|---|---:|---:|---|
| Caltech Faces/Motorbikes, 80 validation images | 100% | 100% | Preserved weights migrated; fresh retinal simulation |
| MNIST official test, 10,000 images | 91.44% | 91.44% | Preserved weights migrated; fresh retinal simulation |
| Caltech, fresh training, seed 42, final epoch 20 | 96.25% | 96.25% | Every plastic pathway trained from scratch |

The first two rows are not scratch-training claims. The fresh Caltech run used
new random kernels and decision weights; fresh MNIST training was not run.
No best-validation-epoch selection was used: epoch 20 is reported.

## Artifacts

- `outputs/framework_converted_caltech/model.pt` and `metrics.json`.
- `outputs/framework_converted_mnist_test/model.pt` and `metrics.json`.
- `outputs/framework_caltech_s42/model.pt` and `metrics.json`.

`evaluation.json` records checkpoint reload, fresh-image evaluation,
checkpoint/source hashes, weight immutability and prediction agreement with
the recorded result.

Caltech uses 200 Faces/Faces_easy and 200 Motorbikes images, split with seed 42
into 320 training and 80 validation examples. Split indices and sampled DoG
preprocessing matched the old pipeline exactly. The fresh run uses 25 11×11
kernels, 5×5 pooling, k=5, 32 sensory ticks, 64 decision ticks per fixation,
batch size 32 and 20 epochs. All nine 35×35 glimpses are freshly simulated.

The old live pipeline defaulted to binding anchor (1,1) but evaluated its grid
at (0,0). The clean pipeline uses (0,0) consistently. Its final seed-42 accuracy
still matches the historical fully plastic run.

## Checks

Migration comparisons established identical sensory outputs/kernel updates,
identical primed spikes with matched anchors, identical decision spike counts
and choices, and R-STDP updates agreeing within floating-point tolerance.
The temporary legacy-dependent tests are retained in the external archive.

Canonical tests cover a shared native graph, changes in every plastic pathway,
selective learning, immutable/repeatable evaluation, actual RF-spike transmission,
checkpoint restoration, and event-integrated eligibility versus explicit local
tick equations.

Every notebook code cell was executed with a small configuration: 8 training
images, 4 validation images, two kernels and one epoch, including CSV/plot cells.
The default sweep was not run. Smoke accuracy is not a benchmark result.

## Preservation

Legacy source, experiments, figures and checkpoints were moved intact to
`../claude_retired_2026-11-09/`; its `data` junction shares the retained dataset.
Earlier winning-model backups remain intact. The pre-refactor archive's 1,385
hashed files were compared with the originals without a mismatch before cleanup.

The failed native-only prototype (42.5% diagnostic accuracy) is outside the
active package. The accepted version ports the working burst/IF dynamics and
centered eligibility rule instead of changing those mechanisms.


## Layout and API refactor — 10 November 2026

Custom behaviors now live in `column/behavior/{neurons,synapse,network,specs}.py`.
Each layer requires the shared `net`; `Column.py` owns interlayer wiring and
canonical `train`/`predict` entrypoints. The CLI delegates training to that API.
The former source layout is backed up in `../claude_before_layout_refactor_2026-11-10/`.

Validation after this refactor:

- Nine tests pass, including all plastic pathways, evaluation immutability,
  checkpoint roundtrip, exact resumed training, and the explicit R-STDP tick oracle.
- A fixture captured before moving the code matches exactly after two training
  minibatches: learned state, inference predictions and decision spike counts.
- Both Caltech checkpoints were re-evaluated on all 80 images: 100% and 96.25%,
  with unchanged weights and exactly matching recorded predictions.
- The first 128 official MNIST test predictions exactly match the recorded result.
  The full 10,000-image evaluation was not repeated for this structural refactor.
- Every notebook code cell executes with 8 training and 4 validation images,
  one epoch, 2 kernels of size 5 and pooling side 2. This is a runnable-code check,
  not an accuracy benchmark or a rerun of the full feature sweep.

Machine-readable evidence and source hashes: `outputs/layout_refactor_validation.json`.
