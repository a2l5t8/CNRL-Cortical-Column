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

from L423.tools.visualize import *

from FC import fullyConnected
from FC.synapse.learning import AttentionBasedRSTDP
from L423.network.SetTarget import *
from L423.L423 import L4, L23
from InputLayer.DataLoaderLayer import DataLoaderLayer
from L56.RefrenceFrames import RefrenceFrame
from InputLayer.synapse.LocationCoder import LocationCoder
from Cortical_Column import NeoCorticalColumn


#######################################################
######################## Config #######################
#######################################################


Input_Width = 28
Input_Height = 28
Crop_Window_Width = 23
Crop_Window_Height = 23
DoG_SIZE = 5

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_NUMBER = 100

OUT_CHANNEL = 8
IN_CHANNEL = 1
KERNEL_WIDTH = 13
KERNEL_HEIGHT = 13

DATASET_IMAGE_WIDTH = IMAGE_WIDTH - DoG_SIZE + 1
DATASET_IMAGE_HEIGHT = IMAGE_HEIGHT - DoG_SIZE + 1
INPUT_WIDTH = Crop_Window_Width - DoG_SIZE + 1
INPUT_HEIGHT = Crop_Window_Height - DoG_SIZE + 1

L4_WIDTH = Crop_Window_Width - KERNEL_WIDTH + 1
L4_HEIGHT = Crop_Window_Height - KERNEL_HEIGHT + 1

L23_WIDTH = L4_WIDTH//2
L23_HEIGHT = L4_HEIGHT//2

J_0 = 300
p = 0.8

iterations = 20000

#######################################################
###################### DataLoader #####################
#######################################################

from torchvision.datasets import MNIST
MNIST_ROOT = "./MNIST"

time_window = 100
crop_iteration = 3

dataset_directory_path = "./first_step"

transformation = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Grayscale(num_output_channels = 1), # not necessary
    Conv2dFilter(DoGFilter(size = 5, sigma_1 = 4, sigma_2 = 1, zero_mean=True, one_sum=True).unsqueeze(0).unsqueeze(0)), # type: ignore
    SqueezeTransform(dim = 0), # type: ignore
])


dataset = MNIST(root=MNIST_ROOT, train=True, download=False, transform=transformation)
first_class = dataset.data[dataset.targets == 4][:40]
second_class = dataset.data[dataset.targets == 9][:40]

target = [0] * len(first_class) + [1] * len(second_class)
target = torch.Tensor(target)

two_class_dataset = torch.cat((first_class, second_class), dim=0)
new_dataset_size = first_class.shape[0] + second_class.shape[0]

t = torch.arange(new_dataset_size)
np.random.shuffle(t.numpy())
two_class_dataset = two_class_dataset[t]
target = target[t]

new_dataset = torch.empty(0, DATASET_IMAGE_WIDTH, DATASET_IMAGE_WIDTH)
centers = []

for i in range(0, new_dataset_size):
    img = two_class_dataset[i]  # 4 in range [0, 5842) ; 9 in range [5842, 11791)
    img = Image.fromarray(img.numpy(), mode="L")
    img = transformation(img)
    # import pdb;pdb.set_trace()
    new_dataset = torch.cat((new_dataset.data, img.data.view(1, *img.data.shape)), dim=0)

dl = DataLoader(new_dataset,shuffle=False)



######################################################
###################### Network #######################
######################################################



net = Neocortex(dt=1, index=True, dtype=torch.float32, behavior = {5 : SetTarget(target = target), 601 : Recorder(["dopamine"])})

#######################################################
####################### Network Layers ################
#######################################################

input_layer = DataLoaderLayer(
    net=net,
    data_loader=dl,
    widnow_size=Crop_Window_Height,
    saccades_on_each_image=7,
    rest_interval=10,
    iterations=iterations
)

L56 = RefrenceFrame(
    net = net, 
    k = 2, 
    refrence_frame_side=28, 
    inhibitory_size=15
)

L4 = L4(net = net, IN_CHANNEL = IN_CHANNEL, OUT_CHANNEL = OUT_CHANNEL, HEIGHT = L4_HEIGHT, WIDTH = L4_WIDTH, INH_SIZE = 7)
L23 = L23(net = net, IN_CHANNEL = IN_CHANNEL, OUT_CHANNEL = OUT_CHANNEL, HEIGHT = L23_HEIGHT, WIDTH = L23_WIDTH)
input_layer = input_layer.build_data_loader()
L56_layer = L56.layer
fclayer = fullyConnected.FC(net = net, N = 100, K = 2)

#######################################################
####################### Connections ###################
#######################################################

Synapsis_L4_L23 = Synapsis(
    net = net,
    src = L4.layer,
    dst = L23.layer,
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
    dst = L4.layer,
    input_port="data_out",
    output_port="input",
    synapsis_behavior=prioritize_behaviors([
        SynapseInit(),
        WeightInitializer(weights = torch.normal(0.1, 2, (OUT_CHANNEL, IN_CHANNEL, KERNEL_HEIGHT, KERNEL_WIDTH)) ),
        Conv2dDendriticInput(current_coef = 20000 , stride = 1, padding = 0),
        Conv2dSTDP(a_plus=0.8, a_minus=0.001),
        WeightNormalization(norm = 4)
    ]),
    synaptic_tag="Proximal"
)


Synapsis_Inp_L56 = Synapsis(
    net = net,
    src = input_layer,
    dst = L56_layer,
    input_port = "data_out",
    output_port = "input",
    synapsis_behavior=prioritize_behaviors([
        SynapseInit(),]) | {
        275: LocationCoder()
    },
    synaptic_tag="Proximal"
)

Synapsis_L23_FC = Synapsis(
    net = net,
    src = L23.layer,
    dst = fclayer.layer,
    input_port="output",
    output_port="input",
    synapsis_behavior=prioritize_behaviors([
        SynapseInit(),
        WeightInitializer(mode = "random"),
        SimpleDendriticInput(current_coef = 50),
        WeightNormalization(norm = 13),
    ]) | ({
        400 : AttentionBasedRSTDP(a_plus = 0.8 , a_minus = 0.1, tau_c = 20, attention_plus = 1.5, attention_minus = -1),
    }),
    synaptic_tag="Proximal"
)

# cc = NeoCorticalColumn()
# inp_to_L4, inp_to_l56 = cc.inject_input(dataset=two_class_dataset, target=target, iterations=iterations)


net.initialize()
net.simulate_iterations(iterations)

#######################################################
####################### Testings ######################
#######################################################

fours = dl.dataset[target == 0]

# to_test_1 = fours[1][1:11, 10:20]
# to_test_2 = fours[2][1:11, 10:20]

# import pdb;pdb.set_trace()

show_filters(Synapsis_Inp_L4.synapses[0].weights)

plt.figure(figsize=(16, 6))
plt.plot(net["dopamine", 0])
plt.show()
