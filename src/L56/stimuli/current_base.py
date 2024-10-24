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
        self, k: int, prob_to_spike: float = 0.5, intensity: float = 0.8, T: int | None = None, *args, **kwargs
    ):
        super().__init__(k=k, prob_to_spike=prob_to_spike, intensity=intensity, T=T, *args, **kwargs)

    def initialize(self, neurons):
        self.k = self.parameter("k", required=True)
        self.intensity = self.parameter("intensity", default=0.8)
        self.prob = self.parameter("prob_to_spike", required=False, default=0.5)
        self.T = self.parameter("T", required=False, default=float("inf"))
        
        self.patterns = [torch.rand(neurons.size) < self.prob for i in range(self.k)]
        
        return super().initialize(neurons)

    def forward(self, neurons):
        neurons.spikes = neurons.vector(0)
        idx = neurons.network.iteration // self.T
        if neurons.network.iteration < self.T * self.k:
            neurons.spikes = torch.logical_and(torch.rand(neurons.size) < self.intensity, self.patterns[idx])
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
