#######################################################
######################## Setup ########################
#######################################################


import torch
from conex import *
from pymonntorch import *

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image

from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader

from conex.helpers.filters import DoGFilter

from tools.visualize import *


#######################################################
######################## Config #######################
#######################################################


DoG_SIZE = 5
IMAGE_WIDTH = 14
IMAGE_HEIGHT = 14

OUT_CHANNEL = 5
IN_CHANNEL = 1
KERNEL_WIDTH = 7
KERNEL_HEIGHT = 7

INPUT_WIDTH = IMAGE_WIDTH - DoG_SIZE + 1
INPUT_HEIGHT = IMAGE_HEIGHT - DoG_SIZE + 1

L4_WIDTH = INPUT_WIDTH - KERNEL_WIDTH + 1
L4_HEIGHT = INPUT_HEIGHT - KERNEL_HEIGHT + 1

L23_WIDTH = L4_WIDTH//2
L23_HEIGHT = L4_HEIGHT//2

J_0 = 300
p = 0.8


#######################################################
###################### DataLoader #####################
#######################################################


time_window = 500

transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels = 1), # not necessary
    Conv2dFilter( DoGFilter(size = 5, sigma_1 = 4, sigma_2 = 1,zero_mean=True,one_sum=True).unsqueeze(0).unsqueeze(0)),
    SqueezeTransform(dim = 0),
    SimplePoisson(time_window = time_window , ratio = 2),
])
dataset = torchvision.datasets.ImageFolder(root="./src/L423/first_step",transform=transformation)
dl = DataLoader(dataset,shuffle=True)


#######################################################
####################### Network #######################
#######################################################



net = Neocortex(dt=1, dtype=torch.float32)



##################### Input Layer #####################



input_layer = InputLayer(
    net=net,
    input_dataloader= dl,
    sensory_data_dim=2,
    sensory_size = NeuronDimension(depth=1, height = INPUT_HEIGHT, width = INPUT_WIDTH),
    sensory_trace= 3,
    instance_duration= time_window,
    silent_interval=50,
    output_ports = {
        "data_out": (None,[("sensory_pop", {})])
    }
)



########################## L4 #########################



ng4e = NeuronGroup(size = NeuronDimension(depth = OUT_CHANNEL , height = L4_HEIGHT, width = L4_WIDTH), net = net, behavior = prioritize_behaviors([
    SimpleDendriteStructure(),
    SimpleDendriteComputation(),
    LIF(
        init_v = -65,
        tau = 7,
        R = 10,
        threshold = -13,
        v_rest = -65,
        v_reset = -70,
    ),
    KWTA(k=20),
    ActivityBaseHomeostasis(window_size=10, activity_rate=200, updating_rate=0.0001),
    Fire(),
    SpikeTrace(tau_s = 15),
    NeuronAxon(),
]))

ng4i = NeuronGroup(size = L4_HEIGHT * L4_WIDTH * OUT_CHANNEL // 4, net = net, tag = "inh", behavior = prioritize_behaviors([
    SimpleDendriteStructure(),
    SimpleDendriteComputation(),
    LIF(
        init_v = -65,
        tau = 7,
        R = 10,
        threshold = -13,
        v_rest = -65,
        v_reset = -70
    ),
    # KWTA(k=30),
    Fire(),
    SpikeTrace(tau_s = 5, offset = 0),
    NeuronAxon(),
]))


sg4e4i = SynapseGroup(net = net, src = ng4e, dst = ng4i, tag = "Proximal", behavior = prioritize_behaviors([
    SimpleDendriticInput(),
    SynapseInit(),
    WeightInitializer(mode = "ones", scale = J_0/math.sqrt(1600 * p), density = 0.02, true_sparsity = False),
]))


sg4i4e = SynapseGroup(net = net, src = ng4i, dst = ng4e, tag = "Proximal", behavior = prioritize_behaviors([
    SimpleDendriticInput(),
    SynapseInit(),
    WeightInitializer(mode = "ones", scale = J_0/math.sqrt(1600 * p), density = 0.02, true_sparsity = False),
]))

sg4e4e = SynapseGroup(net = net, src = ng4e, dst = ng4e, tag = "Proximal", behavior=prioritize_behaviors([
    SynapseInit(),
    WeightInitializer(weights=torch.Tensor([1, 1, 1, 1, 0, 1, 1, 1, 1]).view(1, 1, 9, 1, 1)),
    LateralDendriticInput(current_coef=100000, inhibitory = True),
]))

sg4i4i = SynapseGroup(net = net, src = ng4i, dst = ng4i, tag = "Proximal", behavior = prioritize_behaviors([
    SimpleDendriticInput(),
    SynapseInit(),
    WeightInitializer(mode = "ones", scale = J_0/math.sqrt(1600 * p), density = 0.02, true_sparsity = False),
]))



######################### L2&3 ########################



ng23e = NeuronGroup(size = NeuronDimension(depth = OUT_CHANNEL , height = L23_HEIGHT, width = L23_WIDTH), net = net, behavior = prioritize_behaviors([
    SimpleDendriteStructure(),
    SimpleDendriteComputation(),
    LIF(
        init_v = -65,
        tau = 7,
        R = 10,
        threshold = -13,
        v_rest = -65,
        v_reset = -70
    ),
    Fire(),
    SpikeTrace(tau_s = 15),
    NeuronAxon(),
]))

ng23i = NeuronGroup(size = L23_HEIGHT * L23_WIDTH * OUT_CHANNEL // 4, net = net, tag = "inh", behavior = prioritize_behaviors([
    SimpleDendriteStructure(),
    SimpleDendriteComputation(),
    LIF(
        init_v = -65,
        tau = 7,
        R = 10,
        threshold = -13,
        v_rest = -65,
        v_reset = -70
    ),
    Fire(),
    SpikeTrace(tau_s = 15),
    NeuronAxon(),
]))

sg23e23i = SynapseGroup(net = net, src = ng23e, dst = ng23i, tag = "Proximal", behavior = prioritize_behaviors([
    SimpleDendriticInput(),
    SynapseInit(),
    WeightInitializer(mode = "ones", scale = J_0/math.sqrt(2500 * p), density = 0.02, true_sparsity = False),
]))

sg23i23e = SynapseGroup(net = net, src = ng23i, dst = ng23e, tag = "Proximal", behavior = prioritize_behaviors([
    SimpleDendriticInput(),
    SynapseInit(),
    WeightInitializer(mode = "ones", scale = J_0/math.sqrt(2500 * p), density = 0.02, true_sparsity = False),
]))

sg23i23i = SynapseGroup(net = net, src = ng23i, dst = ng23i, tag = "Proximal", behavior = prioritize_behaviors([
    SimpleDendriticInput(),
    SynapseInit(),
    WeightInitializer(mode = "ones", scale = J_0/math.sqrt(2500 * p), density = 0.02, true_sparsity = False),
]))



######################## Layers #######################



L4 = CorticalLayer(
    net=net,
    excitatory_neurongroup=ng4e,
    inhibitory_neurongroup=ng4i,
    synapsegroups=[sg4e4i, sg4i4e, sg4e4e, sg4i4i],
    input_ports={
        "input": (
            None,
            [Port(object = ng4e, label = None)],
        ),
        "output": (
            None,
            [Port(object = ng4e, label = None)]
        )
    },
)



L23 = CorticalLayer(
    net=net,
    excitatory_neurongroup=ng23e,
    inhibitory_neurongroup=ng23i,
    synapsegroups=[sg23e23i, sg23i23e, sg23i23i],
    input_ports={
        "input": (
            None,
            [Port(object = ng23e, label = None)],
        ),
        "output": (
            None,
            [Port(object = ng23e, label = None)]
        )
    },
)



############### Inter Layer Connections ###############



Synapsis_L4_L23 = Synapsis(
    net = net,
    src = L4,
    dst = L23,
    input_port="output",
    output_port="input",
    synapsis_behavior=prioritize_behaviors([
        SynapseInit(),
        AveragePool2D(current_coef = 50000),
    ]),
    synaptic_tag="Proximal"
)


Synapsis_Inp_L4 = Synapsis(
    net = net,
    src = input_layer,
    dst = L4,
    input_port="data_out",
    output_port="input",
    synapsis_behavior=prioritize_behaviors([
        SynapseInit(),
        WeightInitializer(weights = torch.normal(0.1, 2, (OUT_CHANNEL, IN_CHANNEL, KERNEL_HEIGHT, KERNEL_WIDTH)) ),
        Conv2dDendriticInput(current_coef = 10000 , stride = 1, padding = 0),
        Conv2dSTDP(a_plus=0.3, a_minus=0.008),
        WeightNormalization(norm = 10)
    ]),
    synaptic_tag="Proximal"
)
    


#######################################################
#################### Visualization ####################
#######################################################


net.initialize()
net.simulate_iterations(3000)

show_filters(Synapsis_Inp_L4.synapses[0].weights)