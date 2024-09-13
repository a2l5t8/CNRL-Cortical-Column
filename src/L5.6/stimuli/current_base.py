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

    def initialize(self, neuron):
        self.prob = self.parameter("prob_to_spike", required=False, default=0.5)
        self.T = self.parameter("T", required=False, default=float("inf"))
        return super().initialize(neuron)

    def forward(self, neuron):
        neuron.spikes = neuron.vector(0)
        if neuron.network.iteration < self.T:
            neuron.spikes = torch.rand(neuron.size) < self.prob
        return super().forward(neuron)
