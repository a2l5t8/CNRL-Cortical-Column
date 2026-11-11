# Spiking cortical column

One CoNeX/PymoNNtorch graph with explicit `L4`, `L23`, `L56` and `DecisionLayer`
classes. All dynamics and plasticity execute in framework behaviors. No Layer
containers, GatedRSTDP, Outstar, decision teacher currents or gradient training
are used in the spiking model.

## Verified results

| Evaluation | Previous | New framework |
|---|---:|---:|
| Caltech Faces/Motorbikes, preserved weights, 80 validation images | 100% | **100%** |
| MNIST, preserved weights, 10,000 official test images | 91.44% | **91.44%** |
| Caltech, fresh fully plastic training, seed 42, final epoch 20 | 96.25% | **96.25%** |

The first two rows establish checkpoint migration, not scratch training. The
third starts with random L4 kernels and decision weights and updates every
plastic pathway throughout training. Fresh MNIST training is not claimed.
These are known-position 3Ã—3 classification benchmarks; shifted-object
localization has not been revalidated in this cleaned entrypoint.

See [the diagram and equations](docs/architecture.md) and
[validation/provenance](docs/validation.md).

## Run

Use `requirements.txt` with a CUDA-enabled PyTorch installation. From this directory:

```powershell
python -m pytest tests -q
python -m column.train --dataset caltech --epochs 20 --seed 42 --output outputs/my_caltech_run
python -m column.evaluate outputs/framework_converted_caltech/model.pt
python -m column.evaluate outputs/framework_converted_mnist_test/model.pt --dataset mnist --official-test
```

Training writes `model.pt` and `metrics.json`, with the full learning curve,
final predictions, labels and source hashes. Evaluation writes `evaluation.json`
and verifies that learned weights remain unchanged.

```python
from column import Column, Config

model = Column(Config(maps=25, kernel_size=11, pool=5))
model.net.initialize(info=False, warnings=False)
history = model.train(images, labels, epochs=20, batch_size=32)
prediction = model.predict(images)  # CPU class indices; no learning
model.save("model.pt")
restored = Column.load("model.pt")
```

Inputs are batches of preprocessed 80Ã—80 rate images. Training labels supply
only the authorized reference-frame binding cue and the external scalar reward.
They do not enter the decision-learning rule.

All stages are plastic by default. Use `model.plasticity(('l4',))`,
`('bindings',)` or `('decision',)` to select a stage; `model.plasticity()` enables
all three. Sensory, binding and decision phases are scheduled separately:
all plastic pathways update on arriving training batches, not simultaneously
during every phase.

## Feature-map notebook

[notebooks/feature_map_sweep.ipynb](notebooks/feature_map_sweep.ipynb) includes
imports, data setup, fresh full training, accuracy, CSV output and visualization.
Every code cell was executed with a small smoke configuration; the default full
sweep was not run.

- `FEATURE_COUNTS` / CLI `--maps`: number of learned L4 kernels.
- `KERNEL_SIZES` / `--kernel`: spatial widths, e.g. `[7, 11, 15]`.
- `POOL_SIDES` / `--pool`: pooled map widths, e.g. `[3, 5, 7]`.
- `SEEDS`, `EPOCHS` / `--seed`, `--epochs`: initialization and training duration.

Nine fixed saccades use 35Ã—35 glimpses. L2/3 has `maps Ã— poolÂ²` neurons. Each
configuration learns new features rather than resizing a frozen feature bank.

## Code

- `column/Column.py`: shared network, interlayer wiring, `train`, `predict`, checkpoints.
- `column/L4.py`: retinal input, convolutional L4 and pooling.
- `column/L23.py`: feature population.
- `column/L56.py`: reference-frame population and sensorimotor relays.
- `column/DecisionLayer.py`: excitatory/inhibitory decision circuit.
- `column/behavior/neurons.py`: neuron-group behaviors.
- `column/behavior/synapse.py`: transmission and synaptic learning behaviors.
- `column/behavior/network.py`: network-level reward behavior.
- `column/behavior/specs.py`: current-trial activity storage specs.
- `column/simulation.py`: shared simulation scheduling helpers.
- `column/data/`: splits and DoG preprocessing.
- `column/train.py`, `evaluate.py`: canonical training/evaluation.
- `column/convert.py`: explicit migration of archived weights, not training.
- `baseline/cnn.py`: independent convolution + pooling + MLP benchmark.

Legacy code and artifacts were moved intact to `../claude_retired_2026-11-09/`.
Its `data` junction points to this directory's shared dataset. Earlier winning
model backups also remain intact.

Every layer requires a `net` argument; `Column` supplies the same network to all
four. `train` returns a list of epoch metrics and accepts `validation_data=(x, y)`.
Initialization is automatic in `train` and `predict`, or explicit with
`model.initialize()`. Checkpoints preserve weights, reward baseline, random
states and training history. Existing framework checkpoints remain loadable.
The previous source layout is preserved in `../claude_before_layout_refactor_2026-11-10/`.

## Grid and kernel sweep

From the repository root, run the same command on all four machines and give
each machine a different shard number (1 through 4):

```powershell
python -m sweep --shards 4 --shard 1 --grids 2 3 4 5 6 7 --kernels 3 5 7 9 11
```

With four shards, shard 1 is the server and gets 120 of the 240 default runs.
Shards 2–4 are the laptops and get 40 each. The planner also balances estimated
compute in those proportions. Run IDs stay the same, so gather existing JSON
results into each machine's output directory before resuming under this split.

The default plan uses seed 42. It crosses six
grid sides, five kernel widths, both datasets, and four modes. A grid side of 7
means 49 fixations. Both model families see the same preprocessed crops, with
widths 52, 35, 26, 21, 18, and 15 pixels for the default grids. A kernel must
fit the smallest selected crop. Pooling stays at 5 by 5 in both model families.

- `spiking_rf` uses the grid-specific L5/6 frames, reciprocal STDP, and
  location-bound decision input. `spiking_no_rf` removes L5/6 and uses a
  shared feature-to-decision readout at every fixation. Both train L4 and
  the decision synapse from scratch using STDP and R-STDP.
- `cnn_location` concatenates patch features in fixation order.
  `cnn_no_location` averages the same patch features across fixations.

Use `--dry-run` to preview assignments. Keep the sweep options identical
across machines except for `--shard`, device, and output path. For manual
assignment, use `--shards 1` and give each machine a different subset using
`--grids`, `--kernels`, `--datasets`, or `--modes`. Use `--epochs`,
`--seeds`, `--batch-size`, and `--cnn-batch-size` to set training parameters.

Every finished run writes its own JSON file under
`outputs/grid_kernel_sweep/` (or `--output`), including final validation
accuracy, per-epoch history, and run settings. Relaunch the same command in
the same output directory after an interruption. It skips completed runs
and restarts the interrupted configuration from its seed. Copy the JSON files
from all four machines into one directory to collect the results. Run IDs
include the settings and are independent of shard number.

To try a small run first, use `--shards 1 --grids 3 --kernels 5 --datasets caltech
--epochs 1 --train-limit 8 --val-limit 4 --output outputs/sweep_smoke`.
The full sweep uses the existing validation splits (80 Caltech images and
5,000 MNIST images by default).
