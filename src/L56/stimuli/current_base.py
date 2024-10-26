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
        self, 
        k: int,
        rest_interval: int,    
        prob_to_spike: float = 0.5, 
        intensity: float = 0.8, 
        T: int | None = None, 
        *args, 
        **kwargs
    ):
        super().__init__(k=k, rest_interval=rest_interval, prob_to_spike=prob_to_spike, intensity=intensity, T=T, *args, **kwargs)

    def initialize(self, neurons):
        self.k = self.parameter("k", required=True)
        self.rest_interval = self.parameter("rest_interval", required=True)
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
        elif neurons.network.iteration <= self.T * self.k + self.rest_interval: 
            neurons.spikes = neurons.vector()
        else:
            neurons.spikes = torch.logical_and(torch.rand(neurons.size) < self.intensity, self.patterns[0])
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
    

class PunishModulatorCurrent(pynt.Behavior):
    def __init__(
        self,
        group: str,
        base_line: float,
        punish: float, 
        decay_tau: float, 
        *args, 
        **kwargs
    ):
        super().__init__(group=group,base_line=base_line,punish=punish,decay_tau=decay_tau,*args, **kwargs)
    
    def initialize(self, layer):
        self.group = self.parameter("group", required=True)
        self.base_line = self.parameter("base_line", required=True)
        self.punish = self.parameter("punish", required=True)
        self.decay_tau = self.parameter("decay_tau", required=True)
        self.ngs = []
        for ng in layer.neurongroups:
            if self.group in ng.tags:
                self.ngs.append(ng)
        return super().initialize(layer)
    
    def forward(self, layer):
        spikes = torch.Tensor([torch.sum(ng.spikes, dim=0) for ng in self.ngs])
        winner = torch.argmax(spikes)
        has_winner = torch.max(spikes) != torch.min(spikes)
        for ind, ng in enumerate(self.ngs):
            ng.I += (self.base_line - ng.I)/self.decay_tau + self.punish * (ind != winner) * has_winner
        return super().forward(layer)
