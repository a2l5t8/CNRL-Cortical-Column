# Fully Spiking Cortical Column

This repository now contains only the production fully spiking cortical-column
implementation used by the manuscript experiments. The old gradient-based,
linear-readout, surrogate-readout, cached-memory, and standalone reciprocal
baselines were removed to prevent accidental use of non-paper code paths.

The evaluated model is implemented in CoNeX/PyMoNNtorch:

```text
sensory spikes -> L4 burst spikes -> L2/3 sparse winner spikes
               -> L5/6 location-code LIF spikes
               -> L5/6-gated decision LIF spikes
```

The decision layer is spiking. It learns from free-trial pre/post eligibility
and delayed class-label dopamine; it does not replay target spikes, force output
spikes, use autograd, or attach a classifier on top.

The default training schedule is joint and online from L2/3 onward. Throughout
acquisition, L5/6-to-L2/3 apical plasticity, L2/3-to-L5/6 reciprocal
plasticity, and decision eligibility all operate in the same free multi-saccade
trials. A fixation-gated one-tick binding window prevents a feature from the
previous saccade being assigned to the next location. The class label is
revealed only after the complete trial and affects only decision eligibility
through dopamine. Binding weights are label-independent. Binding plasticity
closes only during the final decision-consolidation epochs.

## Main Source Files

```text
scc/
  fully_spiking.py        CoNeX/PyMoNNtorch fully spiking column
  behaviors/
    neuron.py             custom NeuronGroup behaviors and LIF dynamics
    synapse.py            custom SynapseGroup currents and plasticity
    network.py            network-level behavior namespace (currently empty)
  config.py               dataset/preprocessing config only
  data/
    filters.py            DoG preprocessing
    pipeline.py           MNIST and Caltech loaders
    saccades.py           patch/saccade helpers for diagnostics
  faces_localization.py   non-centered Caltech Faces search helpers

scripts/
  run_fully_spiking_column_experiment.py
  run_fully_spiking_manuscript_experiments.py
  run_multiseed_ablation_study.py
  run_prospective_reference_frame_experiment.py
  run_caltech_faces_localization_experiment.py
  run_caltech_faces_detection_experiment.py
  audit_fully_spiking_checkpoint.py
  build_reproducibility_manifest.py
```

## CoNeX Behavior Usage

The current model uses CoNeX/PyMoNNtorch `Neocortex`, `NeuronGroup`,
`SynapseGroup`, `TimeResolution`, `SynapseInit`, `WeightInitializer`, and
`WeightClip`.

Custom behaviors are retained only where the CoNeX behavior library does not
provide an equivalent batched or location-gated operation:

- deterministic batched sensory phase encoding,
- batched current-driven L5/6 and decision LIF dynamics,
- convolution and pooling over batched spike tensors,
- L5/6-gated decision coincidence current,
- free-trial decision eligibility with delayed class dopamine,
- batched L5/6-to-L2/3 apical coactivity,
- batched L2/3-to-L5/6 reciprocal coactivity.

CoNeX's built-in `LIF`, `KWTA`, `SimpleDendriticInput`, `PreTrace`,
`PostTrace`, and `SimpleRSTDP` are single-sample/general-purpose behaviors.
They are not shape-equivalent to the current batched manuscript runtime, so
replacing the remaining custom behaviors with them would change the model.

## Install

```powershell
pip install -r requirements.txt
python -c "import conex, pymonntorch, torch; print(torch.__version__)"
```

## Run Tests

```powershell
python -m pytest tests -q
```

## Reproduce The Saved Fully Spiking Results

Canonical joint checkpoints are stored under `outputs/joint_training`. The
deterministic L2/3 event caches used to reproduce acquisition remain under
`outputs/best_models`.

```powershell
python scripts\audit_fully_spiking_checkpoint.py `
  --checkpoint outputs\joint_training\caltech_all_acquisition_fixation_gated\best_fully_spiking_column.pt `
  --events outputs\best_models\caltech_fully_spiking_column_no_replay\cortical_spike_events.pt `
  --output outputs\joint_training\caltech_all_acquisition_fixation_gated\fully_spiking_audit.json

python scripts\audit_fully_spiking_checkpoint.py `
  --checkpoint outputs\joint_training\mnist_all_acquisition_fixation_gated\best_fully_spiking_column.pt `
  --events outputs\best_models\mnist_fully_spiking_column_no_replay\cortical_spike_events.pt `
  --output outputs\joint_training\mnist_all_acquisition_fixation_gated\fully_spiking_audit.json
```

## Train Or Re-evaluate

```powershell
python scripts\run_fully_spiking_column_experiment.py `
  --dataset caltech `
  --output-dir outputs\joint_training\caltech_all_acquisition_fixation_gated `
  --events-path outputs\best_models\caltech_fully_spiking_column_no_replay\cortical_spike_events.pt `
  --shuffled-label-control

python scripts\run_fully_spiking_column_experiment.py `
  --dataset mnist `
  --output-dir outputs\joint_training\mnist_all_acquisition_fixation_gated `
  --events-path outputs\best_models\mnist_fully_spiking_column_no_replay\cortical_spike_events.pt `
  --shuffled-label-control
```

These commands use `--training-mode joint`, the default. Pass
`--training-mode staged` only to reproduce the earlier sequential schedule.
L4 kernels remain separately pretrained by convolutional STDP and frozen; the
term joint online training here refers specifically to all trainable apical,
reciprocal, and decision synapses from L2/3 onward. Cached L2/3 spike events are
a deterministic simulation optimization, not a separate classifier or replay
of target activity.

The data loaders automatically download MNIST and Caltech-101 when needed.

## Train Live From Images

These four entry points run the cortical column directly from dataset images.
They do not read or create `cortical_spike_events.pt` files:

```powershell
# Load the saved convolutional-STDP kernels and keep L4 frozen.
python scripts\run_live_caltech_frozen_l4.py
python scripts\run_live_mnist_frozen_l4.py

# Initialize L4 randomly, train it from raw images with local convolutional
# STDP, freeze it, and then train the rest of the column online.
python scripts\run_live_caltech_train_l4.py
python scripts\run_live_mnist_train_l4.py
```

In both regimes, every cortical training sample follows the real `run_images`
path. L5/6-to-L2/3 apical learning, L2/3-to-L5/6 reciprocal learning, and
decision eligibility are active in the same free nine-saccade trial. The class
label produces one delayed dopamine pulse after the complete trial. There is no
target replay or forced decision activity.

The defaults perform the full runs with batch size one. For a short wiring
check, reduce the sample counts explicitly, for example:

```powershell
python scripts\run_live_caltech_train_l4.py `
  --l4-samples 2 --train-samples 2 --eval-samples 2 `
  --l4-epochs 1 --epochs 1 --binding-epochs 1
```

Each run writes `best_fully_spiking_column.pt` and
`live_training_report.json` to its dataset/regime directory under
`outputs/live_online`. The train-L4 scripts additionally write
`learned_l4_weights.pt`.

## Manuscript Bundle

```powershell
python scripts\run_fully_spiking_manuscript_experiments.py `
  --output-dir outputs\manuscript_fully_spiking `
  --device cuda `
  --raw-batch-size 64
```

The manuscript source and compiled paper are in:

```text
manuscript/spiking_cortical_column.tex
manuscript/spiking_cortical_column.pdf
output/pdf/fully_spiking_cortical_column_manuscript.pdf
```

## Current Reference Metrics

From `outputs/manuscript_fully_spiking/manuscript_experiments.json`:

| Dataset | Raw accuracy | Event accuracy | L2/3 to L5/6 top-1 | L2/3 to L5/6 top-3 |
| --- | ---: | ---: | ---: | ---: |
| Caltech Faces/Motorbikes | 1.0000 | 1.0000 | 0.4750 | 0.8111 |
| MNIST | 0.9182 | 0.9198 | 0.3786 | 0.8883 |

Schedule-matched joint training reproduced 1.0000/1.0000 Caltech raw/event
accuracy and 0.9182/0.9198 MNIST raw/event accuracy. MNIST differs by one raw
validation image from the staged checkpoint, while its decision synapse tensors
and event predictions are exactly identical. The selected full-acquisition
reports and checkpoints are in
`outputs/joint_training/caltech_all_acquisition_fixation_gated` and
`outputs/joint_training/mnist_all_acquisition_fixation_gated`.

The reciprocal pathway is phase-gated in the best checkpoints: ordinary object
decisions use `reciprocal_gain = 0.0`, while feature-to-frame assays open the
reciprocal query path with gain 1.0.
