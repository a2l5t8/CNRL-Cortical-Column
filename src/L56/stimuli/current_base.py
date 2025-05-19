import pymonntorch as pynt
import torch


class RandomInputCurrent(pynt.Behavior):
    """
    Randomly forces layer's neurons to trigger spike.

    args:
        prob -> spike triggering probability.
        T    -> specifies iteration to finish the force.
    """

    def __init__(
        self, prob_to_spike: float = 0.5, T: int | None = None, *args, **kwargs
    ):
        super().__init__(prob_to_spike=prob_to_spike, T=T, *args, **kwargs)

    def initialize(self, neurons):
        self.prob = self.parameter("prob_to_spike", required=False, default=0.5)
        self.T = self.parameter("T", required=False, default=float("inf"))
        return super().initialize(neurons)

    def forward(self, neurons):
        neurons.spikes = neurons.vector(0)
        if neurons.network.iteration < self.T:
            neurons.spikes = torch.rand(neurons.size) < self.prob
        return super().forward(neurons)


class ConstantCurrent(pynt.Behavior):
    def __init__(self, scale: float, *args, **kwargs):
        super().__init__(scale = scale,*args, **kwargs)

    def initialize(self, neurons):
        self.scale = self.parameter("scale", 1)
        return super().initialize(neurons)
    
    def forward(self, neurons):
        neurons.I += (self.scale * 1)
        return super().forward(neurons)
