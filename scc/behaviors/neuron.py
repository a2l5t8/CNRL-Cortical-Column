"""Neuron behaviors the cortical column adds to CoNeX."""
import torch
from conex.behaviors.neurons import LIF
from pymonntorch import Behavior

__all__ = ["IF", "MarginWTA"]


class IF(LIF):
    """Non-leaky integrate-and-fire neuron with reset by subtraction.

    tau * dv/dt = R * I; when v >= threshold the neuron spikes and
    v <- v - (threshold - v_reset). CoNeX provides leaky neurons only; without
    leak and with reset by subtraction, a constant input I emits exactly
    floor(T * R * I / (tau * (threshold - v_reset))) spikes in T steps, so spike
    counts carry the integrated input without loss.

    Note: Fire behavior should be added for the neuron to spike.
    """

    def __init__(self, *args, R=1.0, threshold=1.0, tau=1.0, v_reset=0.0, v_rest=0.0, **kwargs):
        super().__init__(R, threshold, tau, v_reset, v_rest, *args, **kwargs)

    def _Fu(self, neurons):
        return 0.0

    def Fire(self, neurons):
        neurons.spikes = neurons.v >= neurons.threshold
        neurons.v[neurons.spikes] -= neurons.threshold - neurons.v_reset


class MarginWTA(Behavior):
    """Winner-take-all with a margin: within each trial a unit spikes while
    v >= max(v) - margin.

    A shared inhibitory interneuron tracks the leading unit and silences every
    unit that trails it by more than ``margin``. Unlike KWTA the number of
    winners is not fixed: it shrinks as evidence separates the units.

    Args:
        margin (float): Largest deficit to the leader that still allows spiking.
    """

    def __init__(self, margin, *args, **kwargs):
        super().__init__(*args, margin=margin, **kwargs)

    def initialize(self, neurons):
        self.margin = self.parameter("margin", None, required=True)
        neurons.spikes = neurons.vector(dtype=torch.bool)

    def forward(self, neurons):
        v = neurons.v.view(neurons.network.batch_size, -1)
        neurons.spikes = (v >= v.amax(1, keepdim=True) - self.margin).view(-1)
