"""End-to-end fully spiking cortical column implemented with CoNeX/PyMoNNtorch.

The runtime deliberately keeps communication between populations event based:

sensory spikes -> L4 burst spikes -> L2/3 winner spikes
               -> L5/6-gated synapses -> decision LIF spikes.

Continuous values are confined to membrane potentials, synaptic weights, and
eligibility traces. The evaluated path never converts an L2/3 activation tensor
into rates for a separate classifier and never selects a decision weight bank
with a Python location index. L5/6 spikes perform the location gating.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from conex import Neocortex, SynapseInit, TimeResolution, WeightClip, WeightInitializer
from pymonntorch import NeuronGroup, SynapseGroup

from scc.behaviors.neuron import (
    BurstingL4Neurons,
    CompetitiveL23Neurons,
    CurrentDrivenL56LIF,
    DecisionLIF,
    MotorCommandSpikes,
    SensoryPhaseEncoder,
)
from scc.behaviors.synapse import (
    BatchedConv2dSTDP,
    DelayedDopaminePlasticity,
    L4ToL23PoolingInput,
    L23ToL56ReciprocalBindingInput,
    L56ApicalFeedbackInput,
    L56GatedDecisionInput,
    MotorToL56Current,
    SpikeTimingEligibility,
    SpikingConvolutionInput,
)


@dataclass
class FullySpikingConfig:
    """Dynamics and plasticity parameters for the complete spiking column."""

    patch_size: int = 35
    feature_map_size: int = 25
    pool_grid_size: int = 5
    l23_k: int = 16
    sensory_steps: int = 16
    l4_release_steps: int = 16
    decision_steps: int = 16
    # Retained only so older checkpoints remain loadable. Decision learning
    # never replays cortical events or forces postsynaptic spikes.
    replay_steps: int = 0
    sensory_rate_scale: float = 1.0
    l4_threshold: float = 1.0
    l4_membrane_decay: float = 1.0
    l4_stdp_learning_rate: float = 0.05
    l4_stdp_a_plus: float = 1.0
    l4_stdp_a_minus: float = 0.1
    l4_stdp_homeostasis: float = 0.05
    l23_membrane_decay: float = 1.0
    decision_membrane_decay: float = 1.0
    decision_threshold: float = 1.0
    decision_synaptic_gain: float = 0.15
    decision_tonic_current: float = 0.25
    decision_learning_rate: float = 0.2
    decision_weight_decay: float = 0.0
    decision_margin: float = 1.0
    eligibility_tau_pre: float = 8.0
    eligibility_tau_post: float = 8.0
    eligibility_tau: float = 128.0
    eligibility_threshold: float = 0.1
    dopamine_tau: float = 5.0
    weight_max: float = 5.0
    apical_learning_rate: float = 0.02
    apical_decay: float = 0.001
    apical_gain: float = 0.05
    apical_max: float = 1.0
    reciprocal_learning_rate: float = 0.2
    reciprocal_decay: float = 0.0
    reciprocal_gain: float = 0.0
    reciprocal_max: float = 1.0
    reciprocal_trace_tau: float = 8.0
    l56_recurrent_gain: float = 1.0
    l56_motor_gain: float = 4.0
    l56_membrane_decay: float = 0.9
    l56_threshold: float = 0.5
    random_state: int = 42


@dataclass
class SpikingBatchResult:
    """Observable event streams produced by one complete column pass."""

    predictions: torch.Tensor
    decision_spike_counts: torch.Tensor
    l23_spike_events: torch.Tensor
    l56_spike_events: torch.Tensor
    sensory_spike_count: torch.Tensor
    l4_spike_count: torch.Tensor
    decision_trace: torch.Tensor


class FullySpikingCorticalColumn:
    """One continuous event-driven cortical-column runtime."""

    backend = "CoNeX/PyMoNNtorch"

    def __init__(
        self,
        n_classes: int,
        n_locations: int,
        l4_kernels: torch.Tensor,
        cfg: Optional[FullySpikingConfig] = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.n_classes = int(n_classes)
        self.n_locations = int(n_locations)
        self.cfg = cfg or FullySpikingConfig()
        self.device = torch.device(device)
        self.l4_kernels = l4_kernels.detach().float().cpu()
        self.n_l4_features = int(l4_kernels.shape[0])
        self.n_l23_features = (
            self.n_l4_features
            * int(self.cfg.pool_grid_size)
            * int(self.cfg.pool_grid_size)
        )
        self.saccade_centers: List[Tuple[int, int]] = []
        self.training_history: List[Dict[str, float]] = []
        self.l4_training_history: List[Dict[str, float]] = []
        self._build()

    def _build(self) -> None:
        self.net = Neocortex(
            dt=1,
            device=str(self.device),
            behavior={1: TimeResolution(dt=1)},
        )
        self.net.cfg = self.cfg
        self.net.batch_size = 1
        self.net.n_locations = self.n_locations
        self.net.n_l4_features = self.n_l4_features
        self.net.cortical_phase = "idle"
        self.net.eligibility_enabled = True
        self.net.reciprocal_learning_enabled = False
        self.net.reciprocal_current_enabled = True
        self.net.sensory_rates = torch.zeros(
            1,
            1,
            self.cfg.patch_size,
            self.cfg.patch_size,
            device=self.device,
        )
        self.net.motor_command_spikes = torch.zeros(
            1, self.n_locations, device=self.device, dtype=torch.bool
        )

        self.sensory_ng = NeuronGroup(
            net=self.net,
            size=self.cfg.patch_size * self.cfg.patch_size,
            behavior={100: SensoryPhaseEncoder()},
            tag="Sensory,Spiking",
        )
        self.motor_ng = NeuronGroup(
            net=self.net,
            size=self.n_locations,
            behavior={100: MotorCommandSpikes()},
            tag="Motor,Spiking",
        )
        self.l56_ng = NeuronGroup(
            net=self.net,
            size=self.n_locations,
            behavior={220: CurrentDrivenL56LIF()},
            tag="L56,ReferenceFrame,LIF,Spiking",
        )
        self.net.l56_ng = self.l56_ng
        self.motor_l56_sg = SynapseGroup(
            net=self.net,
            src=self.motor_ng,
            dst=self.l56_ng,
            behavior={2: SynapseInit(), 180: MotorToL56Current()},
            tag="Proximal,Motor",
        )
        self.net.motor_l56_sg = self.motor_l56_sg

        l4_size = (
            self.n_l4_features
            * self.cfg.feature_map_size
            * self.cfg.feature_map_size
        )
        self.l4_ng = NeuronGroup(
            net=self.net,
            size=l4_size,
            behavior={260: BurstingL4Neurons()},
            tag="L4,Spiking",
        )
        self.sensory_l4_sg = SynapseGroup(
            net=self.net,
            src=self.sensory_ng,
            dst=self.l4_ng,
            behavior={
                2: SynapseInit(),
                180: SpikingConvolutionInput(self.l4_kernels),
                270: BatchedConv2dSTDP(
                    learning_rate=self.cfg.l4_stdp_learning_rate,
                    a_plus=self.cfg.l4_stdp_a_plus,
                    a_minus=self.cfg.l4_stdp_a_minus,
                    homeostasis=self.cfg.l4_stdp_homeostasis,
                ),
            },
            tag="Proximal,Convolutional",
        )
        self.net.sensory_l4_sg = self.sensory_l4_sg

        self.l23_ng = NeuronGroup(
            net=self.net,
            size=self.n_l23_features,
            behavior={300: CompetitiveL23Neurons()},
            tag="L23,Spiking",
        )
        self.l4_l23_sg = SynapseGroup(
            net=self.net,
            src=self.l4_ng,
            dst=self.l23_ng,
            behavior={2: SynapseInit(), 280: L4ToL23PoolingInput()},
            tag="Proximal,Pooling",
        )
        self.net.l4_l23_sg = self.l4_l23_sg
        self.l23_l56_sg = SynapseGroup(
            net=self.net,
            src=self.l23_ng,
            dst=self.l56_ng,
            behavior={
                2: SynapseInit(),
                3: WeightInitializer(
                    mode=0.0,
                    weight_shape=(self.n_locations, self.n_l23_features),
                ),
                181: L23ToL56ReciprocalBindingInput(),
                182: WeightClip(w_min=0.0, w_max=float(self.cfg.reciprocal_max)),
            },
            tag="Reciprocal,FeatureToFrame,Plastic",
        )
        self.net.l23_l56_sg = self.l23_l56_sg
        self.l56_l23_sg = SynapseGroup(
            net=self.net,
            src=self.l56_ng,
            dst=self.l23_ng,
            behavior={
                2: SynapseInit(),
                3: WeightInitializer(
                    mode=0.0,
                    weight_shape=(self.n_locations, self.n_l23_features),
                ),
                281: L56ApicalFeedbackInput(),
                282: WeightClip(w_min=0.0, w_max=float(self.cfg.apical_max)),
            },
            tag="Apical,Predictive,Plastic",
        )
        self.net.l56_l23_sg = self.l56_l23_sg

        self.decision_ng = NeuronGroup(
            net=self.net,
            size=self.n_classes,
            behavior={340: DecisionLIF()},
            tag="Decision,Spiking",
        )
        self.l23_decision_sg = SynapseGroup(
            net=self.net,
            src=self.l23_ng,
            dst=self.decision_ng,
            behavior={
                2: SynapseInit(),
                320: L56GatedDecisionInput(),
                400: SpikeTimingEligibility(),
                420: DelayedDopaminePlasticity(),
            },
            tag="Proximal,L56Gated,Plastic",
        )
        self.net.l23_decision_sg = self.l23_decision_sg

        with contextlib.redirect_stdout(io.StringIO()):
            self.net.initialize(info=False, warnings=False)

    @property
    def effective_decision_weights(self) -> torch.Tensor:
        return (
            self.l23_decision_sg.excitatory_weights
            - self.l23_decision_sg.inhibitory_weights
        )

    def _simulate(self, steps: int = 1) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.net.simulate_iterations(steps, measure_block_time=False)

    def _resize_batch_state(self, batch_size: int) -> None:
        b = int(batch_size)
        self.net.batch_size = b
        dtype = torch.float32
        self.sensory_ng.phase = torch.zeros(
            b, self.sensory_ng.size, device=self.device, dtype=dtype
        )
        self.sensory_ng.spikes = torch.zeros_like(
            self.sensory_ng.phase, dtype=torch.bool
        )
        self.sensory_ng.spike_count = torch.zeros_like(self.sensory_ng.phase)
        self.motor_ng.spikes = torch.zeros(
            b, self.n_locations, device=self.device, dtype=torch.bool
        )
        self.l56_ng.v = torch.zeros(
            b, self.n_locations, device=self.device, dtype=dtype
        )
        self.l56_ng.spikes = torch.zeros_like(self.l56_ng.v, dtype=torch.bool)
        self.l56_ng.spike_count = torch.zeros_like(self.l56_ng.v)
        self.net.active_location_current = torch.zeros(
            b, self.n_locations, device=self.device, dtype=dtype
        )
        self.motor_l56_sg.current = self.net.active_location_current
        self.l4_ng.v = torch.zeros(
            b, self.l4_ng.size, device=self.device, dtype=dtype
        )
        self.l4_ng.spikes = torch.zeros_like(self.l4_ng.v, dtype=torch.bool)
        self.l4_ng.spike_count = torch.zeros_like(self.l4_ng.v)
        self.l23_ng.v = torch.zeros(
            b, self.l23_ng.size, device=self.device, dtype=dtype
        )
        self.l23_ng.spikes = torch.zeros_like(self.l23_ng.v, dtype=torch.bool)
        self.l23_ng.winner_mask = torch.zeros_like(
            self.l23_ng.spikes, dtype=torch.bool
        )
        self.l23_ng.spike_count = torch.zeros_like(self.l23_ng.v)
        self.decision_ng.v = torch.zeros(
            b, self.n_classes, device=self.device, dtype=dtype
        )
        self.decision_ng.spikes = torch.zeros_like(
            self.decision_ng.v, dtype=torch.bool
        )
        self.decision_ng.spike_count = torch.zeros_like(self.decision_ng.v)
        self.l23_decision_sg.pre_trace = torch.zeros(
            b,
            self.n_locations,
            self.n_l23_features,
            device=self.device,
        )
        self.l23_decision_sg.post_trace = torch.zeros(
            b, self.n_classes, device=self.device
        )
        self.l23_decision_sg.eligibility = torch.zeros(
            b,
            self.n_locations,
            self.n_l23_features,
            device=self.device,
        )
        self.l23_decision_sg.post_eligibility = torch.zeros(
            b,
            self.n_classes,
            device=self.device,
        )
        self.l23_l56_sg.current = torch.zeros(
            b, self.n_locations, device=self.device, dtype=dtype
        )
        self.l23_l56_sg.pre_trace = torch.zeros(
            b, self.n_l23_features, device=self.device, dtype=dtype
        )
        self.l23_l56_sg.post_trace = torch.zeros(
            b, self.n_locations, device=self.device, dtype=dtype
        )
        self.l56_l23_sg.pre_trace = torch.zeros(
            b, self.n_locations, device=self.device
        )
        self.l56_l23_sg.post_trace = torch.zeros(
            b, self.n_l23_features, device=self.device
        )
        self.net.motor_command_spikes = torch.zeros(
            b, self.n_locations, device=self.device, dtype=torch.bool
        )
        self.net.gated_pre_spikes = torch.zeros(
            b,
            self.n_locations,
            self.n_l23_features,
            device=self.device,
        )

    def reset_trial(self, batch_size: int) -> None:
        """Reset dynamic neuronal state while preserving learned synapses."""
        self._resize_batch_state(batch_size)
        self.net.cortical_phase = "idle"

    def _extract_patches(
        self,
        images: torch.Tensor,
        centers: Sequence[Tuple[int, int]],
    ) -> List[torch.Tensor]:
        if images.ndim == 4 and images.shape[1] == 1:
            images = images[:, 0]
        if images.ndim != 3:
            raise ValueError("images must have shape (batch, height, width)")
        half = int(self.cfg.patch_size) // 2
        patches: List[torch.Tensor] = []
        for row, col in centers:
            top = int(row) - half
            left = int(col) - half
            patch = images[
                :,
                top:top + self.cfg.patch_size,
                left:left + self.cfg.patch_size,
            ]
            if patch.shape[-2:] != (self.cfg.patch_size, self.cfg.patch_size):
                raise ValueError(
                    f"center {(row, col)} produces patch {tuple(patch.shape[-2:])}"
                )
            patches.append(patch.unsqueeze(1).clamp(0, 1))
        return patches

    def _set_motor_location(self, location: int, pulse: bool = True) -> None:
        command = torch.zeros(
            self.net.batch_size,
            self.n_locations,
            device=self.device,
            dtype=torch.bool,
        )
        if pulse:
            command[:, int(location)] = True
        self.net.motor_command_spikes = command

    def _reset_patch_state(self) -> None:
        self.sensory_ng.phase.zero_()
        self.l4_ng.v.zero_()
        self.l4_ng.spikes.zero_()
        self.l23_ng.v.zero_()
        self.l23_ng.spikes.zero_()
        self.l23_ng.winner_mask.zero_()

    def run_images(
        self,
        images: torch.Tensor,
        centers: Sequence[Tuple[int, int]],
        *,
        learn_apical: bool = False,
        learn_reciprocal: bool = False,
        learn_l4: bool = False,
        track_eligibility: bool = False,
        use_reciprocal: bool = True,
        binding_update_steps: int = 1,
    ) -> SpikingBatchResult:
        """Run raw images through every spiking population."""
        if len(centers) != self.n_locations:
            raise ValueError(
                f"expected {self.n_locations} centers, got {len(centers)}"
            )
        images = images.to(self.device, dtype=torch.float32)
        patches = self._extract_patches(images, centers)
        self.reset_trial(len(images))
        decision_steps = int(self.cfg.decision_steps)
        if decision_steps < 1:
            raise ValueError("decision_steps must be at least one")
        plasticity_steps = min(
            max(int(binding_update_steps), 0),
            max(decision_steps - 1, 0),
        )
        self.net.apical_learning_enabled = False
        self.net.eligibility_enabled = bool(track_eligibility)
        self.net.reciprocal_learning_enabled = False
        self.net.reciprocal_current_enabled = bool(use_reciprocal)
        self.net.l4_learning_enabled = bool(learn_l4)
        l23_events: List[torch.Tensor] = []
        l56_events: List[torch.Tensor] = []
        decision_trace: List[torch.Tensor] = []

        for location, patch in enumerate(patches):
            self._reset_patch_state()
            if learn_apical or learn_reciprocal:
                self.l56_l23_sg.pre_trace.zero_()
                self.l56_l23_sg.post_trace.zero_()
                self.l23_l56_sg.pre_trace.zero_()
                self.l23_l56_sg.post_trace.zero_()
            self.net.sensory_rates = (
                patch * float(self.cfg.sensory_rate_scale)
            ).clamp(0, 1)
            self._set_motor_location(location, pulse=True)
            self.net.cortical_phase = "sensory"
            self._simulate(1)
            self._set_motor_location(location, pulse=False)
            self._simulate(max(int(self.cfg.sensory_steps) - 1, 0))

            self.net.cortical_phase = "l4_release"
            self._simulate(int(self.cfg.l4_release_steps))

            self.net.cortical_phase = "l23_competition"
            self._simulate(1)
            l23_events.append(self.l23_ng.winner_mask.clone())
            l56_events.append(self.l56_ng.spikes.clone())

            if not track_eligibility:
                self.l23_decision_sg.eligibility.zero_()
                self.l23_decision_sg.post_eligibility.zero_()
                self.l23_decision_sg.pre_trace.zero_()
                self.l23_decision_sg.post_trace.zero_()
            self.net.cortical_phase = "decision"
            self._simulate(1)
            remaining_steps = decision_steps - 1
            self.net.apical_learning_enabled = bool(learn_apical)
            self.net.reciprocal_learning_enabled = bool(learn_reciprocal)
            self._simulate(plasticity_steps)
            self.net.apical_learning_enabled = False
            self.net.reciprocal_learning_enabled = False
            self._simulate(remaining_steps - plasticity_steps)
            decision_trace.append(self.decision_ng.spike_count.clone())

        self.net.apical_learning_enabled = False
        self.net.reciprocal_learning_enabled = False
        self.net.l4_learning_enabled = False
        self.net.cortical_phase = "idle"
        counts = self.decision_ng.spike_count.clone()
        return SpikingBatchResult(
            predictions=counts.argmax(dim=1).detach().cpu(),
            decision_spike_counts=counts.detach().cpu(),
            l23_spike_events=torch.stack(l23_events, dim=1).detach().cpu(),
            l56_spike_events=torch.stack(l56_events, dim=1).detach().cpu(),
            sensory_spike_count=self.sensory_ng.spike_count.sum(dim=1).detach().cpu(),
            l4_spike_count=self.l4_ng.spike_count.sum(dim=1).detach().cpu(),
            decision_trace=torch.stack(decision_trace, dim=1).detach().cpu(),
        )

    @torch.no_grad()
    def record_image_dynamics(
        self,
        image: torch.Tensor,
        centers: Sequence[Tuple[int, int]],
    ) -> Dict[str, object]:
        """Record every binary spike emitted during one raw-image trial.

        This diagnostic follows the same scheduler phases as :meth:`run_images`
        and leaves all learned synapses unchanged. It is intentionally limited
        to one image because the full L4 raster can contain tens of millions of
        binary events.
        """
        if len(centers) != self.n_locations:
            raise ValueError(
                f"expected {self.n_locations} centers, got {len(centers)}"
            )
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if image.ndim == 4 and image.shape[1] == 1:
            image = image[:, 0]
        if image.ndim != 3 or len(image) != 1:
            raise ValueError("image must describe exactly one grayscale image")

        image = image.to(self.device, dtype=torch.float32)
        patches = self._extract_patches(image, centers)
        self.reset_trial(1)
        self.net.apical_learning_enabled = False
        self.net.eligibility_enabled = False
        self.net.reciprocal_learning_enabled = False
        self.net.reciprocal_current_enabled = True
        traces: Dict[str, List[torch.Tensor]] = {
            "sensory_spikes": [],
            "l4_spikes": [],
            "l23_spikes": [],
            "l56_spikes": [],
            "decision_spikes": [],
            "l23_membrane": [],
            "decision_membrane": [],
            "decision_spike_count": [],
        }
        phases: List[str] = []
        locations: List[int] = []

        def capture(phase: str, location: int) -> None:
            traces["sensory_spikes"].append(
                self.sensory_ng.spikes[0].detach().cpu().clone()
            )
            traces["l4_spikes"].append(
                self.l4_ng.spikes[0].detach().cpu().clone()
            )
            traces["l23_spikes"].append(
                self.l23_ng.spikes[0].detach().cpu().clone()
            )
            traces["l56_spikes"].append(
                self.l56_ng.spikes[0].detach().cpu().clone()
            )
            traces["decision_spikes"].append(
                self.decision_ng.spikes[0].detach().cpu().clone()
            )
            traces["l23_membrane"].append(
                self.l23_ng.v[0].detach().cpu().clone()
            )
            traces["decision_membrane"].append(
                self.decision_ng.v[0].detach().cpu().clone()
            )
            traces["decision_spike_count"].append(
                self.decision_ng.spike_count[0].detach().cpu().clone()
            )
            phases.append(phase)
            locations.append(int(location))

        def simulate_and_capture(
            phase: str,
            location: int,
            steps: int,
        ) -> None:
            self.net.cortical_phase = phase
            for _ in range(max(int(steps), 0)):
                self._simulate(1)
                capture(phase, location)

        for location, patch in enumerate(patches):
            self._reset_patch_state()
            self.net.sensory_rates = (
                patch * float(self.cfg.sensory_rate_scale)
            ).clamp(0, 1)
            self._set_motor_location(location, pulse=True)
            simulate_and_capture("sensory", location, 1)
            self._set_motor_location(location, pulse=False)
            simulate_and_capture(
                "sensory",
                location,
                int(self.cfg.sensory_steps) - 1,
            )
            simulate_and_capture(
                "l4_release",
                location,
                int(self.cfg.l4_release_steps),
            )
            simulate_and_capture("l23_competition", location, 1)
            simulate_and_capture(
                "decision",
                location,
                int(self.cfg.decision_steps),
            )

        self.net.cortical_phase = "idle"
        result: Dict[str, object] = {
            key: torch.stack(value)
            for key, value in traces.items()
        }
        result.update(
            {
                "phase": phases,
                "location": torch.tensor(locations, dtype=torch.long),
                "patches": torch.cat(
                    [patch.detach().cpu() for patch in patches],
                    dim=0,
                ),
                "prediction": int(
                    self.decision_ng.spike_count[0].argmax().detach().cpu()
                ),
            }
        )
        return result

    def run_spike_events(
        self,
        l23_events: torch.Tensor,
        *,
        track_eligibility: bool = True,
        silence_l56: bool = False,
        use_reciprocal: bool = True,
        learn_apical: bool = False,
        learn_reciprocal: bool = False,
        binding_update_steps: int = 1,
    ) -> SpikingBatchResult:
        """Present cached cortical events to live L5/6 and decision neurons."""
        if l23_events.ndim != 3:
            raise ValueError("l23_events must have shape (batch, locations, features)")
        if tuple(l23_events.shape[1:]) != (
            self.n_locations,
            self.n_l23_features,
        ):
            raise ValueError("unexpected spike-event shape")
        events = l23_events.to(self.device, dtype=torch.bool)
        self.reset_trial(len(events))
        decision_steps = int(self.cfg.decision_steps)
        if decision_steps < 1:
            raise ValueError("decision_steps must be at least one")
        plasticity_steps = min(
            max(int(binding_update_steps), 0),
            max(decision_steps - 1, 0),
        )
        self.net.apical_learning_enabled = False
        self.net.eligibility_enabled = bool(track_eligibility)
        self.net.reciprocal_learning_enabled = False
        self.net.reciprocal_current_enabled = bool(use_reciprocal) and not bool(
            silence_l56
        )
        l56_events: List[torch.Tensor] = []
        decision_trace: List[torch.Tensor] = []
        for location in range(self.n_locations):
            if learn_apical or learn_reciprocal:
                # A new fixation opens a fresh local binding window. Decision
                # eligibility is intentionally not reset and still spans the
                # complete multi-saccade trial.
                self.l56_l23_sg.pre_trace.zero_()
                self.l56_l23_sg.post_trace.zero_()
                self.l23_l56_sg.pre_trace.zero_()
                self.l23_l56_sg.post_trace.zero_()
            l56 = torch.zeros(
                len(events),
                self.n_locations,
                device=self.device,
                dtype=torch.bool,
            )
            if not silence_l56:
                l56[:, location] = True
            # Cached L2/3 events are presented once through the ordinary
            # decision dynamics. This is free inference, not a learning replay:
            # output spikes are generated only by the decision LIF neurons.
            self.l23_ng.winner_mask = events[:, location]
            self.net.motor_command_spikes = l56
            self.net.cortical_phase = "decision"
            # Settle the current saccade before opening local binding
            # plasticity, so traces cannot associate the previous location's
            # residual spikes with the new location.
            self._simulate(1)
            self.net.motor_command_spikes.zero_()
            remaining_steps = decision_steps - 1
            self.net.apical_learning_enabled = bool(learn_apical)
            self.net.reciprocal_learning_enabled = bool(learn_reciprocal)
            self._simulate(plasticity_steps)
            self.net.apical_learning_enabled = False
            self.net.reciprocal_learning_enabled = False
            self._simulate(remaining_steps - plasticity_steps)
            l56_events.append(self.l56_ng.spikes.clone())
            decision_trace.append(self.decision_ng.spike_count.clone())
            if not track_eligibility:
                self.l23_decision_sg.eligibility.zero_()
                self.l23_decision_sg.post_eligibility.zero_()
                self.l23_decision_sg.pre_trace.zero_()
                self.l23_decision_sg.post_trace.zero_()
        self.net.apical_learning_enabled = False
        self.net.reciprocal_learning_enabled = False
        self.net.cortical_phase = "idle"
        counts = self.decision_ng.spike_count.clone()
        return SpikingBatchResult(
            predictions=counts.argmax(dim=1).detach().cpu(),
            decision_spike_counts=counts.detach().cpu(),
            l23_spike_events=events.detach().cpu(),
            l56_spike_events=torch.stack(l56_events, dim=1).detach().cpu(),
            sensory_spike_count=torch.zeros(len(events)),
            l4_spike_count=torch.zeros(len(events)),
            decision_trace=torch.stack(decision_trace, dim=1).detach().cpu(),
        )

    def _reset_eligibility(self) -> None:
        self.l23_decision_sg.pre_trace.zero_()
        self.l23_decision_sg.post_trace.zero_()
        self.l23_decision_sg.eligibility.zero_()
        self.l23_decision_sg.post_eligibility.zero_()

    def _dopamine_pulse(
        self,
        class_pulses: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        self.net.dopamine_pulse = class_pulses.to(
            self.device,
            dtype=torch.float32,
        )
        self.net.dopamine_mask = mask.to(self.device, dtype=torch.bool)
        self.net.cortical_phase = "dopamine"
        self._simulate(1)

    def _apply_outcome_modulation(
        self,
        targets: torch.Tensor,
        competitors: torch.Tensor,
        update_mask: torch.Tensor,
    ) -> None:
        """Modulate eligibility from one free trial without replay or clamping."""
        targets = targets.to(self.device, dtype=torch.long)
        competitors = competitors.to(self.device, dtype=torch.long)
        update_mask = update_mask.to(self.device, dtype=torch.bool)
        pulses = torch.zeros(
            len(targets),
            self.n_classes,
            device=self.device,
            dtype=torch.float32,
        )
        pulses.scatter_(1, targets[:, None], 1.0)
        pulses.scatter_(1, competitors[:, None], -1.0)
        self._dopamine_pulse(
            pulses,
            update_mask,
        )

    def sync_l4_kernels(self) -> torch.Tensor:
        """Copy the live convolutional synapse kernels into model state."""
        self.l4_kernels = (
            self.sensory_l4_sg.raw_kernels.detach().cpu().clone()
        )
        return self.l4_kernels

    def fit_l4_images(
        self,
        loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        centers: Sequence[Tuple[int, int]],
        *,
        epochs: int = 1,
        n_samples: Optional[int] = None,
    ) -> "FullySpikingCorticalColumn":
        """Train convolutional L4 kernels online from raw sensory spikes."""
        self.l4_training_history = []
        initial = self.sync_l4_kernels().clone()
        for epoch in range(int(epochs)):
            seen = 0
            updates_before = int(self.sensory_l4_sg.l4_stdp_update_count)
            for images, _ in loader:
                if n_samples is not None and seen >= int(n_samples):
                    break
                take = len(images)
                if n_samples is not None:
                    take = min(take, int(n_samples) - seen)
                self.run_images(
                    images[:take],
                    centers,
                    learn_l4=True,
                    track_eligibility=False,
                    use_reciprocal=False,
                )
                seen += take
            current = self.sync_l4_kernels()
            self.l4_training_history.append(
                {
                    "epoch": float(epoch + 1),
                    "samples": float(seen),
                    "rms_change_from_initial": float(
                        (current - initial).square().mean().sqrt()
                    ),
                    "stdp_feature_updates": float(
                        int(self.sensory_l4_sg.l4_stdp_update_count)
                        - updates_before
                    ),
                    "winner_coverage": float(
                        (self.sensory_l4_sg.l4_win_count > 0)
                        .float()
                        .mean()
                        .detach()
                        .cpu()
                    ),
                }
            )
        return self

    def fit_joint_images(
        self,
        loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        centers: Sequence[Tuple[int, int]],
        *,
        epochs: int,
        n_samples: Optional[int] = None,
        binding_epochs: Optional[int] = None,
        binding_update_steps: int = 1,
    ) -> "FullySpikingCorticalColumn":
        """Learn both bindings and decision synapses directly from raw images.

        No cortical event tensor is materialized. Each image produces a single
        free multi-saccade trial; feature-location plasticity occurs locally at
        each fixation and label-dependent dopamine is delivered after the
        complete trial.
        """
        total_epochs = int(epochs)
        active_binding_epochs = (
            total_epochs
            if binding_epochs is None
            else min(max(int(binding_epochs), 0), total_epochs)
        )
        self.training_history = []
        for epoch in range(total_epochs):
            seen = 0
            correct = 0
            updated = 0
            binding_active = epoch < active_binding_epochs
            apical_before = float(self.l56_l23_sg.weights.norm().cpu())
            reciprocal_before = int(self.l23_l56_sg.update_count)
            for images, labels in loader:
                if n_samples is not None and seen >= int(n_samples):
                    break
                take = len(images)
                if n_samples is not None:
                    take = min(take, int(n_samples) - seen)
                batch_y = labels[:take].long().cpu()
                result = self.run_images(
                    images[:take],
                    centers,
                    learn_apical=binding_active,
                    learn_reciprocal=binding_active,
                    track_eligibility=True,
                    use_reciprocal=True,
                    binding_update_steps=binding_update_steps,
                )
                counts = result.decision_spike_counts
                predictions = counts.argmax(dim=1)
                target_counts = counts.gather(1, batch_y[:, None]).squeeze(1)
                alternatives = counts.clone()
                alternatives.scatter_(1, batch_y[:, None], -torch.inf)
                competitors = alternatives.argmax(dim=1)
                competitor_counts = counts.gather(
                    1, competitors[:, None]
                ).squeeze(1)
                update_mask = (
                    target_counts - competitor_counts
                    < float(self.cfg.decision_margin)
                )
                correct += int((predictions == batch_y).sum())
                updated += int(update_mask.sum())
                seen += take
                self._apply_outcome_modulation(
                    batch_y,
                    competitors,
                    update_mask,
                )

            if binding_active:
                with torch.no_grad():
                    norm = self.l23_l56_sg.weights.norm(dim=1, keepdim=True)
                    active = norm.squeeze(1) > 0
                    self.l23_l56_sg.weights[active] = (
                        self.l23_l56_sg.weights[active]
                        / norm[active].clamp(min=1e-8)
                    )
                    self.l23_l56_sg.weights.clamp_(
                        0.0,
                        float(self.cfg.reciprocal_max),
                    )

            self.training_history.append(
                {
                    "epoch": float(epoch + 1),
                    "samples": float(seen),
                    "policy_accuracy": correct / max(seen, 1),
                    "update_fraction": updated / max(seen, 1),
                    "learning_replays": 0.0,
                    "forced_output_spikes": 0.0,
                    "raw_image_online": 1.0,
                    "joint_binding_active": float(binding_active),
                    "apical_weight_norm": float(
                        self.l56_l23_sg.weights.norm().detach().cpu()
                    ),
                    "apical_norm_change": float(
                        self.l56_l23_sg.weights.norm().detach().cpu()
                    ) - apical_before,
                    "reciprocal_weight_norm": float(
                        self.l23_l56_sg.weights.norm().detach().cpu()
                    ),
                    "reciprocal_spike_updates": float(
                        int(self.l23_l56_sg.update_count) - reciprocal_before
                    ),
                    "mean_absolute_outcome_modulation": float(
                        self.l23_decision_sg.dopamine_abs.detach().cpu()
                    ),
                }
            )
        return self

    def fit_spike_events(
        self,
        events: torch.Tensor,
        labels: torch.Tensor,
        *,
        epochs: int,
        batch_size: int,
        seed: Optional[int] = None,
    ) -> "FullySpikingCorticalColumn":
        """Train decision synapses with delayed dopamine and event eligibility."""
        events = events.bool().cpu()
        labels = labels.long().cpu()
        generator = torch.Generator().manual_seed(
            int(self.cfg.random_state if seed is None else seed)
        )
        self.training_history = []
        for epoch in range(int(epochs)):
            permutation = torch.randperm(len(events), generator=generator)
            correct = 0
            updated = 0
            for start in range(0, len(events), int(batch_size)):
                indices = permutation[start:start + int(batch_size)]
                batch_x = events[indices]
                batch_y = labels[indices]
                # Eligibility is recorded from this single, unclamped free
                # decision. The label is consulted only after the trial.
                result = self.run_spike_events(batch_x, track_eligibility=True)
                counts = result.decision_spike_counts
                predictions = counts.argmax(dim=1)
                target_counts = counts.gather(1, batch_y[:, None]).squeeze(1)
                alternatives = counts.clone()
                alternatives.scatter_(1, batch_y[:, None], -torch.inf)
                competitors = alternatives.argmax(dim=1)
                competitor_counts = counts.gather(
                    1, competitors[:, None]
                ).squeeze(1)
                update_mask = (
                    target_counts - competitor_counts
                    < float(self.cfg.decision_margin)
                )
                correct += int((predictions == batch_y).sum())
                updated += int(update_mask.sum())

                # Target and competitor modulation acts on eligibility already
                # produced by their natural free-trial spikes. No event or
                # postsynaptic spike is replayed, injected, or clamped.
                self._apply_outcome_modulation(
                    batch_y,
                    competitors,
                    update_mask,
                )

            self.training_history.append(
                {
                    "epoch": float(epoch + 1),
                    "policy_accuracy": correct / max(len(events), 1),
                    "update_fraction": updated / max(len(events), 1),
                    "learning_replays": 0.0,
                    "forced_output_spikes": 0.0,
                    "dopamine": float(
                        self.l23_decision_sg.dopamine.detach().cpu()
                    ),
                    "mean_absolute_outcome_modulation": float(
                        self.l23_decision_sg.dopamine_abs.detach().cpu()
                    ),
                }
            )
        return self

    def fit_joint_spike_events(
        self,
        events: torch.Tensor,
        labels: torch.Tensor,
        *,
        epochs: int,
        batch_size: int,
        binding_epochs: Optional[int] = None,
        binding_update_steps: int = 1,
        seed: Optional[int] = None,
    ) -> "FullySpikingCorticalColumn":
        """Train both binding pathways and decision synapses in free trials.

        L5/6-to-L2/3 and L2/3-to-L5/6 plasticity uses only the naturally
        coincident spikes at each saccade. Decision eligibility accumulates in
        the same trial, and the class label is consulted only after all
        saccades to create the delayed dopamine pulse.
        """
        events = events.bool().cpu()
        labels = labels.long().cpu()
        total_epochs = int(epochs)
        active_binding_epochs = (
            total_epochs
            if binding_epochs is None
            else min(max(int(binding_epochs), 0), total_epochs)
        )
        generator = torch.Generator().manual_seed(
            int(self.cfg.random_state if seed is None else seed)
        )
        self.training_history = []
        for epoch in range(total_epochs):
            permutation = torch.randperm(len(events), generator=generator)
            correct = 0
            updated = 0
            binding_active = epoch < active_binding_epochs
            apical_updates_before = float(self.l56_l23_sg.weights.norm().cpu())
            reciprocal_count_before = int(self.l23_l56_sg.update_count)
            for start in range(0, len(events), int(batch_size)):
                indices = permutation[start:start + int(batch_size)]
                batch_x = events[indices]
                batch_y = labels[indices]
                result = self.run_spike_events(
                    batch_x,
                    track_eligibility=True,
                    learn_apical=binding_active,
                    learn_reciprocal=binding_active,
                    binding_update_steps=binding_update_steps,
                )
                counts = result.decision_spike_counts
                predictions = counts.argmax(dim=1)
                target_counts = counts.gather(1, batch_y[:, None]).squeeze(1)
                alternatives = counts.clone()
                alternatives.scatter_(1, batch_y[:, None], -torch.inf)
                competitors = alternatives.argmax(dim=1)
                competitor_counts = counts.gather(
                    1, competitors[:, None]
                ).squeeze(1)
                update_mask = (
                    target_counts - competitor_counts
                    < float(self.cfg.decision_margin)
                )
                correct += int((predictions == batch_y).sum())
                updated += int(update_mask.sum())
                self._apply_outcome_modulation(
                    batch_y,
                    competitors,
                    update_mask,
                )

            if binding_active:
                with torch.no_grad():
                    norm = self.l23_l56_sg.weights.norm(dim=1, keepdim=True)
                    active = norm.squeeze(1) > 0
                    self.l23_l56_sg.weights[active] = (
                        self.l23_l56_sg.weights[active]
                        / norm[active].clamp(min=1e-8)
                    )
                    self.l23_l56_sg.weights.clamp_(
                        0.0,
                        float(self.cfg.reciprocal_max),
                    )

            self.training_history.append(
                {
                    "epoch": float(epoch + 1),
                    "policy_accuracy": correct / max(len(events), 1),
                    "update_fraction": updated / max(len(events), 1),
                    "learning_replays": 0.0,
                    "forced_output_spikes": 0.0,
                    "dopamine": float(
                        self.l23_decision_sg.dopamine.detach().cpu()
                    ),
                    "mean_absolute_outcome_modulation": float(
                        self.l23_decision_sg.dopamine_abs.detach().cpu()
                    ),
                    "joint_binding_active": float(binding_active),
                    "apical_weight_norm": float(
                        self.l56_l23_sg.weights.norm().detach().cpu()
                    ),
                    "apical_norm_change": float(
                        self.l56_l23_sg.weights.norm().detach().cpu()
                    ) - apical_updates_before,
                    "reciprocal_weight_norm": float(
                        self.l23_l56_sg.weights.norm().detach().cpu()
                    ),
                    "reciprocal_spike_updates": float(
                        int(self.l23_l56_sg.update_count)
                        - reciprocal_count_before
                    ),
                }
            )
        return self

    def fit_apical_spike_events(
        self,
        events: torch.Tensor,
        *,
        epochs: int = 1,
        batch_size: int = 256,
    ) -> "FullySpikingCorticalColumn":
        """Learn predictive L5/6-to-L2/3 synapses from spike coactivity."""
        events = events.bool().cpu()
        for _ in range(int(epochs)):
            for start in range(0, len(events), int(batch_size)):
                batch = events[start:start + int(batch_size)].to(self.device)
                self.reset_trial(len(batch))
                self.net.apical_learning_enabled = True
                self.net.eligibility_enabled = False
                self.net.reciprocal_learning_enabled = False
                self.net.reciprocal_current_enabled = False
                for location in range(self.n_locations):
                    self.l23_ng.winner_mask = batch[:, location]
                    command = torch.zeros(
                        len(batch),
                        self.n_locations,
                        device=self.device,
                        dtype=torch.bool,
                    )
                    command[:, location] = True
                    self.net.motor_command_spikes = command
                    self.net.cortical_phase = "decision"
                    # First step establishes L5/6 and L2/3 spikes; the second
                    # performs the local pre/post apical STDP update.
                    self._simulate(1)
                    self.net.motor_command_spikes.zero_()
                    self._simulate(1)
        self.net.apical_learning_enabled = False
        self.net.cortical_phase = "idle"
        return self

    def fit_reciprocal_spike_events(
        self,
        events: torch.Tensor,
        *,
        epochs: int = 1,
        batch_size: int = 256,
        shuffle_targets: bool = False,
        seed: Optional[int] = None,
    ) -> "FullySpikingCorticalColumn":
        """Learn L2/3-to-L5/6 feature-frame synapses from spike coactivity."""
        if events.ndim != 3:
            raise ValueError("events must have shape (samples, locations, features)")
        if tuple(events.shape[1:]) != (self.n_locations, self.n_l23_features):
            raise ValueError("unexpected spike-event shape")
        cues = events.bool().reshape(-1, self.n_l23_features).cpu()
        targets = torch.arange(self.n_locations).repeat(len(events))
        generator = torch.Generator().manual_seed(
            int(self.cfg.random_state if seed is None else seed)
        )
        if shuffle_targets:
            targets = targets[
                torch.randperm(len(targets), generator=generator)
            ]
        for _ in range(int(epochs)):
            order = torch.randperm(len(cues), generator=generator)
            for start in range(0, len(cues), int(batch_size)):
                index = order[start:start + int(batch_size)]
                batch = cues[index].to(self.device)
                location = targets[index].to(self.device, dtype=torch.long)
                self.reset_trial(len(batch))
                self.net.apical_learning_enabled = False
                self.net.eligibility_enabled = False
                self.net.reciprocal_learning_enabled = True
                self.net.reciprocal_current_enabled = False
                self.l23_ng.winner_mask = batch
                command = F.one_hot(
                    location,
                    num_classes=self.n_locations,
                ).to(self.device, dtype=torch.bool)
                self.net.motor_command_spikes = command
                self.net.cortical_phase = "decision"
                # First tick establishes motor-driven L5/6 spikes and L2/3 cue
                # spikes. The second tick lets the local pre/post trace update
                # observe those spikes without replaying any sensory event.
                self._simulate(1)
                self.net.motor_command_spikes.zero_()
                self._simulate(1)
            with torch.no_grad():
                norm = self.l23_l56_sg.weights.norm(dim=1, keepdim=True)
                active = norm.squeeze(1) > 0
                self.l23_l56_sg.weights[active] = (
                    self.l23_l56_sg.weights[active]
                    / norm[active].clamp(min=1e-8)
                )
                self.l23_l56_sg.weights.clamp_(
                    0.0,
                    float(self.cfg.reciprocal_max),
                )
        self.net.reciprocal_learning_enabled = False
        self.net.reciprocal_current_enabled = True
        self.net.cortical_phase = "idle"
        return self

    @torch.no_grad()
    def predict_l56_from_l23_events(
        self,
        cues: torch.Tensor,
        *,
        batch_size: int = 512,
        steps: int = 6,
        gain: Optional[float] = None,
        record_first: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Retrieve L5/6 locations using only L2/3 spike cues."""
        if cues.ndim == 3:
            cues = cues.reshape(-1, cues.shape[-1])
        if cues.ndim != 2 or cues.shape[1] != self.n_l23_features:
            raise ValueError("cues must have shape (trials, features)")
        original_gain = float(self.cfg.reciprocal_gain)
        if gain is not None:
            self.cfg.reciprocal_gain = float(gain)
        predictions: List[torch.Tensor] = []
        spike_counts: List[torch.Tensor] = []
        currents: List[torch.Tensor] = []
        first_raster = None
        try:
            for start in range(0, len(cues), int(batch_size)):
                batch = cues[start:start + int(batch_size)].bool().to(self.device)
                self.reset_trial(len(batch))
                self.net.apical_learning_enabled = False
                self.net.eligibility_enabled = False
                self.net.reciprocal_learning_enabled = False
                self.net.reciprocal_current_enabled = True
                self.net.motor_command_spikes.zero_()
                self.net.active_location_current.zero_()
                self.l23_ng.winner_mask = batch
                self.l23_ng.spikes = batch.clone()
                self.net.cortical_phase = "decision"
                raster: List[torch.Tensor] = []
                for _ in range(int(steps)):
                    self._simulate(1)
                    if record_first and first_raster is None:
                        raster.append(
                            self.l56_ng.spikes[0].detach().cpu().clone()
                        )
                if raster:
                    first_raster = torch.stack(raster)
                predictions.append(
                    self.l56_ng.spike_count.argmax(dim=1).detach().cpu()
                )
                spike_counts.append(self.l56_ng.spike_count.detach().cpu())
                currents.append(self.l23_l56_sg.current.detach().cpu())
        finally:
            if gain is not None:
                self.cfg.reciprocal_gain = original_gain
            self.net.cortical_phase = "idle"
        result = {
            "predictions": torch.cat(predictions),
            "spike_counts": torch.cat(spike_counts),
            "currents": torch.cat(currents),
        }
        if first_raster is not None:
            result["first_raster"] = first_raster
        return result

    @torch.no_grad()
    def evaluate_spike_events(
        self,
        events: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 256,
    ) -> Dict[str, object]:
        predictions: List[torch.Tensor] = []
        counts: List[torch.Tensor] = []
        for start in range(0, len(events), int(batch_size)):
            result = self.run_spike_events(
                events[start:start + int(batch_size)],
                track_eligibility=False,
            )
            predictions.append(result.predictions)
            counts.append(result.decision_spike_counts)
        pred = torch.cat(predictions)
        targets = labels.long().cpu()
        return {
            "accuracy": float((pred == targets).float().mean()),
            "predictions": pred,
            "targets": targets,
            "decision_spike_counts": torch.cat(counts),
        }

    @torch.no_grad()
    def evaluate_images(
        self,
        loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        centers: Sequence[Tuple[int, int]],
        n_samples: int,
    ) -> Dict[str, object]:
        predictions: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        seen = 0
        for images, labels in loader:
            if seen >= int(n_samples):
                break
            remaining = int(n_samples) - seen
            images = images[:remaining]
            labels = labels[:remaining]
            result = self.run_images(images, centers, track_eligibility=False)
            predictions.append(result.predictions)
            targets.append(labels.long().cpu())
            seen += len(images)
        pred = torch.cat(predictions)
        target = torch.cat(targets)
        return {
            "accuracy": float((pred == target).float().mean()),
            "predictions": pred,
            "targets": target,
        }

    def checkpoint(self, metadata: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        self.sync_l4_kernels()
        return {
            "kind": "fully_spiking_cortical_column",
            "backend": self.backend,
            "gradient_free": True,
            "surrogate_derivatives": False,
            "decision_learning_rule": "online_free_trial_three_factor",
            "reciprocal_binding_rule": "local_l23_l56_pre_post_coactivity",
            "learning_replays": False,
            "forced_output_spikes": False,
            "n_classes": self.n_classes,
            "n_locations": self.n_locations,
            "n_l4_features": self.n_l4_features,
            "n_l23_features": self.n_l23_features,
            "cfg": asdict(self.cfg),
            "l4_kernels": self.l4_kernels,
            "decision_excitatory_weights": (
                self.l23_decision_sg.excitatory_weights.detach().cpu()
            ),
            "decision_inhibitory_weights": (
                self.l23_decision_sg.inhibitory_weights.detach().cpu()
            ),
            "apical_feedback_weights": (
                self.l56_l23_sg.weights.detach().cpu()
            ),
            "reciprocal_binding_weights": (
                self.l23_l56_sg.weights.detach().cpu()
            ),
            "training_history": self.training_history,
            "l4_training_history": self.l4_training_history,
            "metadata": metadata or {},
        }

    def load_checkpoint(self, checkpoint: Dict[str, object]) -> None:
        if checkpoint.get("kind") != "fully_spiking_cortical_column":
            raise ValueError("not a fully-spiking cortical-column checkpoint")
        with torch.no_grad():
            self.l23_decision_sg.excitatory_weights.copy_(
                checkpoint["decision_excitatory_weights"].to(self.device)
            )
            self.l23_decision_sg.inhibitory_weights.copy_(
                checkpoint["decision_inhibitory_weights"].to(self.device)
            )
            self.l56_l23_sg.weights.copy_(
                checkpoint["apical_feedback_weights"].to(self.device)
            )
            if "reciprocal_binding_weights" in checkpoint:
                self.l23_l56_sg.weights.copy_(
                    checkpoint["reciprocal_binding_weights"].to(self.device)
                )
            else:
                self.l23_l56_sg.weights.zero_()
        self.training_history = list(checkpoint.get("training_history", []))
        self.l4_training_history = list(
            checkpoint.get("l4_training_history", [])
        )


def load_fully_spiking_column(
    path: str,
    device: torch.device | str = "cpu",
) -> FullySpikingCorticalColumn:
    checkpoint = torch.load(path, map_location="cpu")
    cfg = FullySpikingConfig(**checkpoint["cfg"])
    model = FullySpikingCorticalColumn(
        n_classes=int(checkpoint["n_classes"]),
        n_locations=int(checkpoint["n_locations"]),
        l4_kernels=checkpoint["l4_kernels"],
        cfg=cfg,
        device=device,
    )
    model.load_checkpoint(checkpoint)
    return model
