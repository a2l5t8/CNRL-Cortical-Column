import pymonntorch as pynt
from conex import *


class vDistributor(pynt.Behavior):
    def initialize(self, synapse):
        super().initialize(synapse)
    def forward(self, synapse):
        import pdb;pdb.set_trace()

