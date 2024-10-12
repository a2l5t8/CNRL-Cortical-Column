import torch
from conex import *
from pymonntorch import *

class SetTarget(Behavior) :

    def initialize(self, network) : 
        network.network_target = self.parameter("target", required = True)
        network.targets = network.network_target[0]

    def forward(self, network) :
        # network.targets = self.network_target[network.iteration]
        network.dopamine = torch.Tensor([network.dopamine_concentration])