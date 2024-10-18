#######################################################
######################## Setup ########################
#######################################################

import random
import torch
from conex import *
from pymonntorch import *
import tqdm

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image

import torchvision
from torch.utils.data import DataLoader

from conex.helpers.filters import DoGFilter

from src.L423.tools.visualize import *

from src.FC import fullyConnected
from src.FC.synapse.learning import AttentionBasedRSTDP
from src.L423.network.SetTarget import *
from src.L423.L423 import L4, L23, SensoryLayer
from src.InputLayer.DataLoaderLayer import DataLoaderLayer
from src.L56.RefrenceFrames import RefrenceFrame
from src.InputLayer.synapse.LocationCoder import LocationCoder

#######################################################
######################## Config #######################
#######################################################


Input_Width = 28
Input_Height = 28
Crop_Window_Width = 21
Crop_Window_Height = 21
DoG_SIZE = 5

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_NUMBER = 100

OUT_CHANNEL = 8
IN_CHANNEL = 1
KERNEL_WIDTH = 13
KERNEL_HEIGHT = 13

INPUT_WIDTH = IMAGE_WIDTH - DoG_SIZE + 1
INPUT_HEIGHT = IMAGE_HEIGHT - DoG_SIZE + 1
# INPUT_WIDTH = Crop_Window_Width - DoG_SIZE + 1
# INPUT_HEIGHT = Crop_Window_Height - DoG_SIZE + 1

L4_WIDTH = INPUT_WIDTH - KERNEL_WIDTH + 1
L4_HEIGHT = INPUT_HEIGHT - KERNEL_HEIGHT + 1

L23_WIDTH = L4_WIDTH//2
L23_HEIGHT = L4_HEIGHT//2

J_0 = 300
p = 0.8

# net = Neocortex(dt = 1)

# sensory = SensoryLayer(
#     net=net,
# )
# rf = RefrenceFrame(net = net)

# cc_connection_L4_L23 = CorticalLayerConnection(
#     net = net,
#     src = L4.layer,
#     dst = L23.layer,
#     synapsis_behavior=prioritize_behaviors([
#         SynapseInit(),
#         AveragePool2D(current_coef = 50000),
#     ]),
#     synaptic_tag="Proximal"
# )

# cc_connection_L23_L56 = CorticalLayerConnection(
#     net=net,
#     src = L4
# )



# Synapsis_Inp_L4 = CorticalLayerConnection(
#     net = net,
#     src = input_layer,
#     dst = L4.layer,
#     synapsis_behavior=prioritize_behaviors([
#         SynapseInit(),
#         WeightInitializer(weights = torch.normal(0.1, 2, (OUT_CHANNEL, IN_CHANNEL, KERNEL_HEIGHT, KERNEL_WIDTH)) ),
#         Conv2dDendriticInput(current_coef = 20000 , stride = 1, padding = 0),
#         Conv2dSTDP(a_plus=0.3, a_minus=0.0008),
#         WeightNormalization(norm = 4)
#     ]),
#     synaptic_tag="Proximal"
# )


# Synapsis_Inp_L56 = Synapsis(
#     net = net,
#     src = input_layer,
#     dst = L56_layer,
#     input_port = "data_out",
#     output_port = "input",
#     synapsis_behavior=prioritize_behaviors([
#         SynapseInit(),]) | {
#         275: LocationCoder()
#     },
#     synaptic_tag="Proximal"
# )

# cc = CorticalColumn(
#     net = net,
#     layers={
#         "L4" : sensory.L4,
#         "L23" : sensory.L23,
#         "L56" : rf.layer,
#     },
#     layer_connections = [
#         ("L4", "L23", cc_connection_L4_L23),
#         ("L23", "L56", cc_connection_L23_L56),
#         ("L56", "L23", cc_connection_L56_L23),
#     ],
#     input_ports = {
#         "sensory_input" : (None, Port(object = sensory.L4, label = "input")),
#         "location_input" : (None, Port(object = rf.layer, label = "input")),
#     },
#     output_ports = {
#         "sensory_output" : (None, Port(object = sensory.L23, label = "output")),
#         "location_output" : (None, Port(object = rf.layer, label = "output")),
#     }
# )


# synapsis_sensory_input_cc = Synapsis(
#     net = net,
#     src = input_layer,
#     dst = cc,
#     input_port = "data_out",
#     output_port = "sensnory_input",
#     synapsis_behavior=prioritize_behaviors([
#         SynapseInit(),
#         WeightInitializer(weights = torch.normal(0.1, 2, (OUT_CHANNEL, IN_CHANNEL, KERNEL_HEIGHT, KERNEL_WIDTH)) ),
#         Conv2dDendriticInput(current_coef = 10000 , stride = 1, padding = 0),
#         Conv2dSTDP(a_plus=0.3, a_minus=0.008),
#         WeightNormalization(norm = 10)
#     ]),
#     synaptic_tag="Proximal"
# )

class NeoCorticalColumn():
    def __init__(
        self,
        net: Neocortex = None,
    ):  
        ### in test
        target = [0] * 10 + [1] * 10
        target = torch.Tensor(target)
        ###
        
        self.net = net
        if not self.net:
            self.net = Neocortex(
                dt=1, 
                index=True, 
                dtype=torch.float32, 
                behavior = prioritize_behaviors
                (
                        [
                            Payoff(initial_payoff = 1),
                            Dopamine(tau_dopamine = 5),
                        ]
                ) | {5 : SetTarget(target = target), 601 : Recorder(["dopamine"])}
            )

        ### layers
        
        self.L56 = RefrenceFrame(
            net = self.net, 
            k = 2, 
            refrence_frame_side=28, 
            inhibitory_size=15
        ).layer

        self.L4 = L4(net = self.net, IN_CHANNEL = IN_CHANNEL, OUT_CHANNEL = OUT_CHANNEL, HEIGHT = L4_HEIGHT, WIDTH = L4_WIDTH, INH_SIZE = 7).layer
        self.L23 = L23(net = self.net, IN_CHANNEL = IN_CHANNEL, OUT_CHANNEL = OUT_CHANNEL, HEIGHT = L23_HEIGHT, WIDTH = L23_WIDTH).layer   

        ### connections
        cc_connection_L23_L56 = ("L23", "L56", self._L23_L56_cc_connection())
        cc_connection_L56_L23 = ("L56", "L23", self._L56_L23_cc_connection())
        cc_connection_L4_L23 = ("L4", "L23", self._L4_L23_cc_connection())
        
        ### cortica layer
        self.cc = CorticalColumn(
            net=self.net,
            layers={
                "L4" : self.L4,
                "L23" : self.L23,
                "L56" : self.L56,
            },
            layer_connections = [
                # ("L4", "L23", cc_connection_L4_L23),
                # ("L23", "L56", cc_connection_L23_L56),
                # ("L56", "L23", cc_connection_L56_L23),
                cc_connection_L23_L56,
                cc_connection_L4_L23,
                cc_connection_L56_L23
            ],
            input_ports = {
                "sensory_input" : (None, Port(object = self.L4, label = "input")),
                "location_input" : (None, Port(object = self.L56, label = "input")),
            },
            output_ports = {
                "sensory_output" : (None, Port(object = self.L23, label = "output")),
                "location_output" : (None, Port(object = self.L56, label = "output")),
            }
        )
        
    def _L23_L56_cc_connection(self):
        clc = CorticalLayerConnection(net=self.net)
        l23 = self.L23.neurongroups[0] if "inh" in self.L23.neurongroups[1].tags else self.L23.neurongroups[1]
        for ng in self.L56.neurongroups:
            if 'RefrenceFrame' in ng.tags:
                sg = SynapseGroup(
                    net=self.net,
                    src=l23,
                    dst=ng,
                    tag="Apical, exi",
                    behavior=prioritize_behaviors(
                        [SynapseInit(), SimpleDendriticInput() ,WeightInitializer(mode="normal(0.2, 3)")]
                    )
                )
                clc.synapses.append(sg)
        clc.src = self.L23
        clc.dst = self.L56
        return clc
    
    def _L56_L23_cc_connection(self):
        clc = CorticalLayerConnection(net=self.net)
        l23 = self.L23.neurongroups[0] if "inh" in self.L23.neurongroups[1].tags else self.L23.neurongroups[1]
        for ng in self.L56.neurongroups:
            if 'RefrenceFrame' in ng.tags:
                sg = SynapseGroup(
                    net=self.net,
                    src=ng,
                    dst=l23,
                    tag="Apical, exi",
                    behavior=prioritize_behaviors(
                        [SynapseInit(), SimpleDendriticInput() ,WeightInitializer(mode="normal(0.2, 3)")]
                    )
                )
                clc.synapses.append(sg)
        clc.src = self.L56
        clc.dst = self.L23
        return clc

    def _L4_L23_cc_connection(self):
        clc = CorticalLayerConnection(net=self.net)
        l23 = self.L23.neurongroups[0] if "inh" in self.L23.neurongroups[1].tags else self.L23.neurongroups[1]
        l4 = self.L4.neurongroups[0] if "inh" in self.L4.neurongroups[1].tags else self.L4.neurongroups[1]
        sg = SynapseGroup(
            net=self.net,
            src=l4,
            dst=l23,
            behavior=prioritize_behaviors([
                SynapseInit(),
                WeightInitializer(weights = torch.normal(0.1, 2, (OUT_CHANNEL, IN_CHANNEL, KERNEL_HEIGHT, KERNEL_WIDTH)) ),
                Conv2dDendriticInput(current_coef = 20000 , stride = 1, padding = 0),
                Conv2dSTDP(a_plus=0.3, a_minus=0.0008),
                WeightNormalization(norm = 4)
            ]),
            tag="Proximal"
        )
        clc.synapses.append(sg)
        clc.src = self.L4
        clc.dst = self.L23
        return clc
        
            

# net.initialize()
# net.simulate_iterations(3000)