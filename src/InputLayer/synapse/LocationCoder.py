import pymonntorch as pynt
from conex import *


class LocationCoder(pynt.Behavior):
    """
        note: The src neuron group should either have OnlineDataLoader of focus_loc. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self, synapse):
        synapse.dst.last_position = synapse.src.focus_loc
        synapse.dst._v = torch.tensor([0, 0])
        synapse.I = synapse.matrix().view(-1)
        return super().initialize(synapse)

    def is_locs_equal(self, synapse):
        return (synapse.src.focus_loc[0] == self.last_position[0] and synapse.src.focus_loc[1] == self.last_position[1])

    def forward(self, synapse):
        if synapse.src.focus_loc[0] == -1:
            synapse.dst._v = torch.tensor([torch.nan, torch.nan])
            return super().forward(synapse)
    
        synapse.dst._v[0] = synapse.src.focus_loc[0] - synapse.dst.last_position[0]
        synapse.dst._v[1] = synapse.src.focus_loc[1] - synapse.dst.last_position[1]
        synapse.dst.last_position = synapse.src.focus_loc
        # print("-->",synapse.network.iteration, ":",synapse.tags, synapse.dst.tags, synapse.dst._v, self.last_position)
        return super().forward(synapse)
