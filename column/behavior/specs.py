"""Current-trial state specifications and activity traces."""

import torch
from pymonntorch import Behavior


class PoolTrace(Behavior):
    def initialize(self, neurons):
        batch = neurons.network.batch_size
        if getattr(neurons.network, "reset_pool_trace", True) or not hasattr(
            neurons, "held_current"
        ):
            neurons.held_current = torch.zeros(
                batch,
                getattr(neurons.network, "fixations", 9),
                neurons.size // batch,
                device=neurons.device,
            )

    def forward(self, neurons):
        net = neurons.network
        if getattr(net, "capture_pool", False):
            neurons.held_current[:, net.slot] = neurons.v.reshape(net.batch_size, -1)


class SpikeTrace(Behavior):
    def initialize(self, neurons):
        batch = neurons.network.batch_size
        neurons.held_spikes = torch.zeros(
            batch,
            getattr(neurons.network, "fixations", 9),
            neurons.size // batch,
            device=neurons.device,
            dtype=torch.bool,
        )

    def forward(self, neurons):
        net = neurons.network
        if getattr(net, "capture_context", False):
            neurons.held_spikes[:, net.slot] = neurons.spikes.reshape(
                net.batch_size, -1
            )
