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
from src.L423.L423 import L4, L23
from src.InputLayer.DataLoaderLayer import DataLoaderLayer


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

iterations = 1000

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
    # SimplePoisson(time_window = time_window , ratio = 2), # type: ignore
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

# target_new = [[i] * (time_window * crop_iteration + 100) for i in target]
# target_new = torch.Tensor(target_new)
# target = target_new.view(-1)

new_dataset = torch.empty(0, INPUT_HEIGHT, INPUT_WIDTH)
centers = []

# def confidence_crop_interspace(inp_width, inp_height, window_width, window_height):
#     x1 = window_width//2
#     x2 = (inp_width - 1) - (window_width//2)
#     y1 = window_height//2 
#     y2 = (inp_height - 1) - (window_height//2)

#     center_x = random.randint(x1, x2)
#     center_y = random.randint(y1, y2)
#     center_coordinates = [center_x, center_y]
#     top_left_x = center_x - (window_width//2)
#     top_left_y = center_y - (window_height//2)
#     top_left_coordinates = [top_left_x, top_left_y]
#     coordinates = [center_coordinates, top_left_coordinates]

#     return coordinates

for i in range(0, new_dataset_size):
    img = two_class_dataset[i]  # 4 in range [0, 5842) ; 9 in range [5842, 11791)
    img = Image.fromarray(img.numpy(), mode="L")
    # a = confidence_crop_interspace(Input_Width, Input_Height, Crop_Window_Width, Crop_Window_Height)
    # centers.append((a[0][0], a[0][1]))
    # cropped_image = torchvision.transforms.functional.crop(img, a[1][1], a[1][0], Crop_Window_Width, Crop_Window_Height)
    # # cropped_image = Image.fromarray(cropped_image.numpy(), mode="L")
    # # cropped_image = img
    img = transformation(img)
    # cropped_image = cropped_image.view(time_window, Crop_Window_Width - DoG_SIZE + 1, Crop_Window_Height - DoG_SIZE + 1)
    new_dataset = torch.cat((new_dataset.data, img.data.view(1, *img.data.shape)), dim=0)

# new_dataset = new_dataset.view((new_dataset_size, crop_iteration * time_window, INPUT_WIDTH, INPUT_HEIGHT))
# print(new_dataset.shape)

dl = DataLoader(new_dataset,shuffle=False)



#######################################################
####################### Network #######################
#######################################################



net = Neocortex(dt=1, dtype=torch.float32, behavior = prioritize_behaviors(
    [
        Payoff(initial_payoff = 1),
        Dopamine(tau_dopamine = 5),
    ]
    ) | {5 : SetTarget(target = target), 601 : Recorder(["dopamine"])})


# input_layer = InputLayer(
#     net=net,
#     input_dataloader = dl,
#     sensory_data_dim=2,
#     sensory_size = NeuronDimension(depth=1, height = INPUT_HEIGHT, width = INPUT_WIDTH),
#     sensory_trace = 3,
#     instance_duration = time_window,
#     have_label = False,
#     silent_interval = 100,
#     output_ports = {
#         "data_out": (None,[("sensory_pop", {})])
#     }
# )

input_layer = DataLoaderLayer(
    net=net,
    data_loader=dl,
    widnow_size=INPUT_HEIGHT,
    saccades_on_each_image=5,
    rest_interval=100,
    iterations=iterations
)
L4 = L4(net = net, IN_CHANNEL = IN_CHANNEL, OUT_CHANNEL = OUT_CHANNEL, HEIGHT = L4_HEIGHT, WIDTH = L4_WIDTH, INH_SIZE = 7)
L23 = L23(net = net, IN_CHANNEL = IN_CHANNEL, OUT_CHANNEL = OUT_CHANNEL, HEIGHT = L23_HEIGHT, WIDTH = L23_WIDTH)


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

input_layer = input_layer.build_data_loader()

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
        Conv2dSTDP(a_plus=0.3, a_minus=0.0008),
        WeightNormalization(norm = 4)
    ]),
    synaptic_tag="Proximal"
)

net.initialize()
net.simulate_iterations(iterations)

import pdb;pdb.set_trace()

show_filters(Synapsis_Inp_L4.synapses[0].weights)