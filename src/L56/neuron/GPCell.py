import torch

import pymonntorch as pynt
from conex import LIF


class GPCell(LIF):
    """
    Works as both L6 & L5 layers toghether as Grid-Place Cell getting speed vector as input.

    Args :
        (float) L : The movement scaler, used as a scaler for the amount of effect the speed vector has on GPCells movement.
        (tuple) V : Speed-Vector throughout the iterations.
        (float) I_amp : Constant injected amplitude current.
    """

    def __init__(
        self,
        R,
        threshold,
        tau,
        v_reset,
        v_rest,
        *args,
        init_v=None,
        init_s=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            R=R,
            tau=tau,
            threshold=threshold,
            v_reset=v_reset,
            v_rest=v_rest,
            init_v=init_v,
            init_s=init_s,
            **kwargs,
        )

    def initialize(self, neurons):
        super().initialize(neurons)

        self.L = self.parameter("L", 1)
        self.V = self.parameter("V", required=True)
        self.I_amp = self.parameter("I_amp", 5)
        self.shape = (neurons.shape.depth, neurons.shape.width, neurons.shape.height)

        neurons.spike_prev = neurons.vector("zeros") < 0

    def forward(self, neurons):
        # newPosX = (
        #     neurons.x[neurons.spike_prev]
        #     + (self.shape[1] - 1) / 2
        #     + self.V[0][neurons.network.iteration] * self.L
        # )

        # newPosY = (
        #     neurons.y[neurons.spike_prev]
        #     + (self.shape[2] - 1) / 2
        #     + self.V[1][neurons.network.iteration] * self.L
        # )
        newPosX = (
            neurons.x[neurons.spike_prev]
            + (self.shape[1] - 1) / 2
            + neurons._v[0] * self.L
        )

        newPosY = (
            neurons.y[neurons.spike_prev]
            + (self.shape[2] - 1) / 2
            + neurons._v[1] * self.L
        )

        newPosX = torch.round(newPosX).to(dtype=torch.int32) % self.shape[1]
        newPosY = torch.round(newPosY).to(dtype=torch.int32) % self.shape[2]

        if newPosX.numel() and newPosY.numel():
            neurons.I.view(self.shape[2], self.shape[1])[newPosY, newPosX] += self.I_amp

        super().forward(neurons)

    def Fire(self, neurons):
        super().Fire(neurons)
        neurons.spike_prev = neurons.spikes
