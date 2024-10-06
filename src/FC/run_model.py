from pymonntorch import *
from conex import *
import torch

import matplotlib.pyplot as plt

from FC import FC


net = Network(behavior = prioritize_behaviors([
    TimeResolution(dt = 1)
]))

fclayer = FC(K = 5, N = 50, net = net)

input_ng = NeuronGroup(
    net = net,
    size = 100,
    behavior = prioritize_behaviors([
        SimpleDendriteStructure(),
        SimpleDendriteComputation(),
        LIF(
            R = 10,
            tau = 5,
            threshold = -10,
            v_rest = -65,
            v_reset = -67,
            init_v =  -65,
        ),
        # InherentNoise(scale=random.randint(20, 60)),
        Fire(),
        NeuronAxon()
        ]) | ({ 
            600 : Recorder(["I"]),
            601 : EventRecorder(['spikes'])
        }),
    tag = "exi"
)

input_layer = Layer(
    net = net,
    neurongroups = [input_ng],
    synapsegroups = [],
    output_ports = {
        "output" : (
            None,
            [Port(object = input_ng)]
        )
    }
)

print(input_layer)
fclayer.create_input_connection(input_layer)

print(fclayer.layer)

# net.initialize()
# net.simulate_iterations(100)

# for i in range(fclayer.K) : 
#     plt.plot(fclayer.net.NeuronGroups[i]['spikes.t', 0], fclayer.net.NeuronGroups[i]['spikes.i', 0] + (i * 50), '.')
# plt.show()