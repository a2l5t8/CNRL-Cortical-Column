"""Custom behaviors attached to cortical-column neuron groups."""
from __future__ import annotations

import torch
from pymonntorch import Behavior, NeuronGroup

__all__ = [
    "SensoryPhaseEncoder",
    "MotorCommandSpikes",
    "CurrentDrivenL56LIF",
    "BurstingL4Neurons",
    "CompetitiveL23Neurons",
    "DecisionLIF",
]


class SensoryPhaseEncoder(Behavior):
    """Deterministic phase encoder that emits binary retinal spike events."""

    def initialize(self, neurons: NeuronGroup) -> None:
        super().initialize(neurons)
        neurons.phase = torch.zeros(
            1, neurons.size, device=neurons.device, dtype=neurons.def_dtype
        )
        neurons.spikes = torch.zeros_like(neurons.phase, dtype=torch.bool)
        neurons.spike_count = torch.zeros_like(neurons.phase)

    def forward(self, neurons: NeuronGroup) -> None:
        net = neurons.network
        if net.cortical_phase != "sensory":
            neurons.spikes = torch.zeros_like(neurons.spikes)
            return
        rates = net.sensory_rates.reshape(net.batch_size, -1)
        neurons.phase = neurons.phase + rates
        neurons.spikes = neurons.phase >= 1.0
        neurons.phase = neurons.phase - neurons.spikes.to(neurons.phase.dtype)
        neurons.spike_count = neurons.spike_count + neurons.spikes


class MotorCommandSpikes(Behavior):
    """Expose an externally supplied one-hot saccade command as spikes."""

    def initialize(self, neurons: NeuronGroup) -> None:
        super().initialize(neurons)
        neurons.spikes = torch.zeros(
            1, neurons.size, device=neurons.device, dtype=torch.bool
        )

    def forward(self, neurons: NeuronGroup) -> None:
        command = getattr(neurons.network, "motor_command_spikes", None)
        if command is None:
            neurons.spikes = torch.zeros_like(neurons.spikes)
        else:
            neurons.spikes = command.to(neurons.device, dtype=torch.bool)


class CurrentDrivenL56LIF(Behavior):
    """LIF location neurons driven by a sustained one-hot input current."""

    def initialize(self, neurons: NeuronGroup) -> None:
        super().initialize(neurons)
        neurons.v = torch.zeros(
            1, neurons.size, device=neurons.device, dtype=neurons.def_dtype
        )
        neurons.spikes = torch.zeros_like(neurons.v, dtype=torch.bool)
        neurons.spike_count = torch.zeros_like(neurons.v)

    def forward(self, neurons: NeuronGroup) -> None:
        net = neurons.network
        reciprocal = (
            net.l23_l56_sg.current
            if getattr(net, "reciprocal_current_enabled", True)
            else torch.zeros_like(net.motor_l56_sg.current)
        )
        neurons.v = (
            float(net.cfg.l56_membrane_decay) * neurons.v
            + net.motor_l56_sg.current
            + float(net.cfg.reciprocal_gain) * reciprocal
        )
        neurons.spikes = neurons.v >= float(net.cfg.l56_threshold)
        neurons.v = torch.where(
            neurons.spikes,
            torch.zeros_like(neurons.v),
            neurons.v,
        )
        neurons.spike_count = neurons.spike_count + neurons.spikes


class BurstingL4Neurons(Behavior):
    """Integrate sensory spike current, then emit subtractive-reset LIF bursts."""

    def initialize(self, neurons: NeuronGroup) -> None:
        super().initialize(neurons)
        neurons.v = torch.zeros(
            1, neurons.size, device=neurons.device, dtype=neurons.def_dtype
        )
        neurons.spikes = torch.zeros_like(neurons.v, dtype=torch.bool)
        neurons.spike_count = torch.zeros_like(neurons.v)

    def forward(self, neurons: NeuronGroup) -> None:
        net = neurons.network
        phase = net.cortical_phase
        if phase == "sensory":
            neurons.v = (
                float(net.cfg.l4_membrane_decay) * neurons.v
                + net.sensory_l4_sg.current
            )
            neurons.spikes = torch.zeros_like(neurons.spikes)
        elif phase == "l4_release":
            neurons.spikes = neurons.v >= float(net.cfg.l4_threshold)
            neurons.v = neurons.v - (
                neurons.spikes.to(neurons.v.dtype) * float(net.cfg.l4_threshold)
            )
            neurons.spike_count = neurons.spike_count + neurons.spikes
        else:
            neurons.spikes = torch.zeros_like(neurons.spikes)


class CompetitiveL23Neurons(Behavior):
    """Integrate proximal/apical input and emit sparse winner spikes."""

    def initialize(self, neurons: NeuronGroup) -> None:
        super().initialize(neurons)
        neurons.v = torch.zeros(
            1, neurons.size, device=neurons.device, dtype=neurons.def_dtype
        )
        neurons.spikes = torch.zeros_like(neurons.v, dtype=torch.bool)
        neurons.winner_mask = torch.zeros_like(neurons.spikes)
        neurons.spike_count = torch.zeros_like(neurons.v)

    def forward(self, neurons: NeuronGroup) -> None:
        net = neurons.network
        phase = net.cortical_phase
        if phase == "l4_release":
            proximal = net.l4_l23_sg.current
            apical = torch.tanh(net.l56_l23_sg.current)
            neurons.v = (
                float(net.cfg.l23_membrane_decay) * neurons.v
                + proximal
                + float(net.cfg.apical_gain) * apical
            )
            neurons.spikes = torch.zeros_like(neurons.spikes)
            return

        if phase == "l23_competition":
            k = min(int(net.cfg.l23_k), neurons.size)
            _, indices = torch.topk(neurons.v, k=k, dim=1)
            neurons.winner_mask = torch.zeros_like(neurons.spikes)
            neurons.winner_mask.scatter_(1, indices, True)
            neurons.spikes = neurons.winner_mask.clone()
        elif phase == "decision":
            neurons.spikes = neurons.winner_mask.clone()
        else:
            neurons.spikes = torch.zeros_like(neurons.spikes)
        neurons.spike_count = neurons.spike_count + neurons.spikes


class DecisionLIF(Behavior):
    """Thresholded decision LIF population with subtractive reset."""

    def initialize(self, neurons: NeuronGroup) -> None:
        super().initialize(neurons)
        neurons.v = torch.zeros(
            1, neurons.size, device=neurons.device, dtype=neurons.def_dtype
        )
        neurons.spikes = torch.zeros_like(neurons.v, dtype=torch.bool)
        neurons.spike_count = torch.zeros_like(neurons.v)

    def forward(self, neurons: NeuronGroup) -> None:
        net = neurons.network
        if net.cortical_phase != "decision":
            neurons.spikes = torch.zeros_like(neurons.spikes)
            return

        neurons.v = (
            float(net.cfg.decision_membrane_decay) * neurons.v
            + net.l23_decision_sg.current
            + float(net.cfg.decision_tonic_current)
        )
        spikes = neurons.v >= float(net.cfg.decision_threshold)
        neurons.spikes = spikes
        neurons.v = neurons.v - (
            spikes.to(neurons.v.dtype) * float(net.cfg.decision_threshold)
        )
        neurons.spike_count = neurons.spike_count + neurons.spikes

