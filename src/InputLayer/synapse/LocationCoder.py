import pymonntorch as pynt
from conex import *


class LocationCoder(pynt.Behavior):
    """
        note: The src neuron group should either have OnlineDataLoader of focus_loc. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, synapse):
        self.last_position = synapse.src.focus_loc
        synapse.dst._v = torch.tensor([0, 0])
        return super().initialize(synapse)

    def is_locs_equal(self, synapse):
        return (synapse.src.focus_loc[0] == self.last_position[0] and synapse.src.focus_loc[1] == self.last_position[1])

    def forward(self, synapse):

        synapse.dst._v[0] = synapse.src.focus_loc[0] - self.last_position[0]
        synapse.dst._v[1] = synapse.src.focus_loc[1] - self.last_position[1]
        self.last_position = synapse.src.focus_loc
        return super().forward(synapse)
