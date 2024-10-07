import torch

from pymonntorch import *
from conex import *

from L423.L423 import SensoryLayer
from L56 import RefrenceFrames


net = Neocortex(dt = 1)

sensory = SensoryLayer(net = net)
rf = ReferenceFrame(net = net)

cc = CorticalColumn(
    net = net,
    layers={
        "L4" : sensory.L4,
        "L23" : sensory.L23,
        "L56" : rf.layer,
    },
    layer_connections = [
        ("L4", "L23", cc_connection_L4_L23),
        ("L23", "L56", cc_connection_L23_L56),
        ("L56", "L23", cc_connection_L56_L23),
    ],
    input_ports = {
        "sensory_input" : (None, Port(object = sensory.L4, label = "input")),
        "location_input" : (None, Port(object = rf.layer, label = "input")),
    },
    output_ports = {
        "sensory_output" : (None, Port(object = sensory.L23, label = "output")),
        "location_output" : (None, Port(object = rf.layer, label = "output")),
    }
)


synapsis_sensory_input_cc = Synapsis(
    net = net,
    src = input_layer,
    dst = cc,
    input_port = "data_out",
    output_port = "sensnory_input",
    synapsis_behavior=prioritize_behaviors([
        SynapseInit(),
        WeightInitializer(weights = torch.normal(0.1, 2, (OUT_CHANNEL, IN_CHANNEL, KERNEL_HEIGHT, KERNEL_WIDTH)) ),
        Conv2dDendriticInput(current_coef = 10000 , stride = 1, padding = 0),
        Conv2dSTDP(a_plus=0.3, a_minus=0.008),
        WeightNormalization(norm = 10)
    ]),
    synaptic_tag="Proximal"
)

net.initialize()
net.simulate_iterations(3000)