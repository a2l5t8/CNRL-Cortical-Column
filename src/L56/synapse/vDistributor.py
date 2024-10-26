from typing import Sequence, Tuple

import pymonntorch as pynt
from conex import *



class ManualVCoder(pynt.Behavior):
    def __init__(
        self,
        points: Sequence[Tuple[int, int]],
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
    
    def initialize(self, synapse):
        self.points = self.parameter("points", required=True)
        synapse.dst._v = torch.tensor([0, 0])
        self.offset = synapse.network.iteration
        super().initialize(synapse)
    
    def forward(self, synapse):
        synapse.dst._v[0] = self.points[synapse.network.iteration - self.offset][0] - synapse.dst.last_position[0]
        synapse.dst._v[1] = self.points[synapse.network.iteration - self.offset][1] - synapse.dst.last_position[1]
        synapse.dst.last_position = self.points[synapse.network.iteration - self.offset]

