# Fully Spiking Cortical Column: Reproducibility Notes

Document state: 2026-07-10  
Canonical seed: 42  
Primary framework: CoNeX 0.1.5 / PyMoNNtorch 0.1.4  

This repository has been refactored to keep only the fully spiking model used
by the current manuscript. Legacy gradient-based, linear-readout,
surrogate-readout, cached-memory, and standalone reciprocal baselines were
removed from the source tree.

## Source Of Truth

The production model is:

- `scc/fully_spiking.py`

Supporting code:

- `scc/config.py`: dataset and preprocessing configuration.
- `scc/data/pipeline.py`: MNIST and Caltech loaders.
- `scc/data/filters.py`: Difference-of-Gaussians preprocessing.
- `scc/data/saccades.py`: small saccade/patch diagnostics.
- `scc/faces_localization.py`: uncropped Caltech Faces search helpers.

## Model Summary

The complete evaluated path is spiking:

```text
retinal/sensory spikes
  -> L4 convolutional burst spikes
  -> L2/3 sparse winner spikes
  -> L5/6 location-code LIF spikes
  -> L5/6-gated decision LIF spikes
```

The decision layer receives coincident L2/3 and L5/6 spikes through a physical
CoNeX/PyMoNNtorch `SynapseGroup`. Learning uses free-trial spike eligibility
and a delayed class-label dopamine pulse. The label is consulted after the
trial only to decide reward/punishment; no target spikes are replayed or
forced.

The model also contains two physical feature-location pathways:

- L5/6 -> L2/3 apical feedback for frame-to-feature priming.
- L2/3 -> L5/6 reciprocal feedback for feature-to-frame retrieval.

## Joint Online Plasticity

The canonical training command now uses joint plasticity from L2/3 onward.
Throughout acquisition, both feature-location pathways learn from the same
natural L2/3 and L5/6 spikes that drive decision eligibility. A new fixation
first establishes the current location and feature spikes and then opens a one-step local
binding window. The short apical and reciprocal traces reset at fixation
boundaries; the decision eligibility trace is not reset and spans all nine
saccades. Dopamine is delivered only after the final saccade.

Binding plasticity closes only for the final dopamine-gated decision
consolidation epochs. This is a stability gate, not a separate offline replay:
no cortical event or output spike is regenerated, injected, or clamped. The
class label cannot affect either binding direction. Unit tests verify
label-invariant binding weights and bit-identical decision weights between
joint and schedule-matched decision-only training when reciprocal
classification gain is zero.

L4 kernels are still pretrained separately with convolutional STDP and remain
fixed. Thus "joint online" refers to every trainable pathway from L2/3 onward,
not to joint optimization of L4.

The best classification checkpoints keep reciprocal gain at zero during normal
object decisions and open the reciprocal path only during the feature-to-frame
assay.

## CoNeX Behavior Usage

The implementation uses CoNeX/PyMoNNtorch `Neocortex`, `NeuronGroup`,
`SynapseGroup`, `TimeResolution`, `SynapseInit`, `WeightInitializer`, and
`WeightClip`.

Custom behaviors remain only where no built-in behavior is shape-equivalent:

- batched sensory phase encoding,
- batched L5/6 and decision LIF dynamics,
- batched L4 convolution and L2/3 pooling currents,
- L5/6-gated decision coincidence current,
- delayed-dopamine eligibility plasticity,
- batched apical and reciprocal coactivity updates.

CoNeX's built-in `LIF`, `KWTA`, `SimpleDendriticInput`, `PreTrace`,
`PostTrace`, and `SimpleRSTDP` operate on the standard single-sample behavior
interface and are not direct replacements for those batched manuscript
operations.

## Canonical Artifacts

Canonical joint checkpoints:

- Caltech:
  `outputs/joint_training/caltech_all_acquisition_fixation_gated/best_fully_spiking_column.pt`
- MNIST:
  `outputs/joint_training/mnist_all_acquisition_fixation_gated/best_fully_spiking_column.pt`

Cached cortical event tensors:

- Caltech:
  `outputs/best_models/caltech_fully_spiking_column_no_replay/cortical_spike_events.pt`
- MNIST:
  `outputs/best_models/mnist_fully_spiking_column_no_replay/cortical_spike_events.pt`

Manuscript metrics:

- `outputs/manuscript_fully_spiking/manuscript_experiments.json`

Manuscript source:

- `manuscript/spiking_cortical_column.tex`
- `manuscript/spiking_cortical_column.pdf`

## Canonical Metrics

From `outputs/manuscript_fully_spiking/manuscript_experiments.json`:

| Dataset | Raw accuracy | Event accuracy | Reciprocal top-1 | Reciprocal top-3 | Shuffled reciprocal top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Caltech Faces/Motorbikes | 1.0000 | 1.0000 | 0.4750 | 0.8111 | 0.1028 |
| MNIST | 0.9182 | 0.9198 | 0.3786 | 0.8883 | 0.1146 |

Full schedule-matched joint runs produced:

| Dataset | Joint raw accuracy | Joint event accuracy | Current-ranked reciprocal top-1 | Current-ranked reciprocal top-3 |
| --- | ---: | ---: | ---: | ---: |
| Caltech Faces/Motorbikes | 1.0000 | 1.0000 | 0.4847 | 0.8250 |
| MNIST | 0.9182 | 0.9198 | 0.5079 | 0.9008 |

The MNIST event result and decision tensors are exactly equal to the staged
checkpoint. Raw accuracy differs by one image among 5,000 held-out samples.
The selected reports and checkpoints are:

- `outputs/joint_training/caltech_all_acquisition_fixation_gated`
- `outputs/joint_training/mnist_all_acquisition_fixation_gated`

## Verification Commands

Run unit tests:

```powershell
python -m pytest tests -q
```

Audit saved checkpoints:

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

Rebuild the manuscript experiment bundle:

```powershell
python scripts\run_fully_spiking_manuscript_experiments.py `
  --output-dir outputs\manuscript_fully_spiking `
  --device cuda `
  --raw-batch-size 64
```
