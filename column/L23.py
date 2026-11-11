"""L2/3 feature neurons with proximal drive and apical priming."""

from conex import (
    KWTA,
    LIF,
    Fire,
    NeuronAxon,
    SimpleDendriteComputation,
    SimpleDendriteStructure,
    prioritize_behaviors,
)
from pymonntorch import NeuronGroup

from .behavior.neurons import BindingClamp
from .behavior.specs import SpikeTrace


class L23:
    def __init__(self, net, features, winners):
        self.net = net
        behaviors = prioritize_behaviors(
            [
                SimpleDendriteStructure(Apical_max_delay=1, apical_min_delay=0),
                SimpleDendriteComputation(apical_provocativeness=0.5),
                LIF(R=1, tau=1, threshold=12.5, v_reset=0, v_rest=0),
                KWTA(k=winners, dimension=1),
                Fire(),
                NeuronAxon(),
            ]
        )
        behaviors.update({350: BindingClamp(), 360: SpikeTrace()})
        self.neurons = NeuronGroup(
            net=net, size=features, tag="L23", behavior=behaviors
        )
        self.objects = [self.neurons]
