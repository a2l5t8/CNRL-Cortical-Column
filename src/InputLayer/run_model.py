# #### -- PROJECT PATH -- ####
# import sys
# sys.path.append(r'C:\Users\amilion\Documents\GitHub\CNRL-Cortical-Column')

#### -- IMPORTS -- ####

from matplotlib import pyplot as plt


from pymonntorch import *
from conex import *

### IMPORTS
import torch
from conex import *
from pymonntorch import *

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image

from tqdm import tqdm

from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader

from conex.helpers.filters import DoGFilter


from src.InputLayer.stimuli.OnlineDataLoader import OnlineDataLoader
from src.InputLayer.synapse.LocationCoder import LocationCoder

from src.L56.neuron.GPCell import GPCell
from src.L56.stimuli.current_base import ConstantCurrent, RandomInputCurrent
from src.L56.synapse.GPCell_lateral_inhibition import GPCellLateralInhibition

from src.L56.tools.rat_simulation import speed_vector_converter, generate_walk
from src.L56.tools.visualization import iter_spike_multi_real

from torchvision.datasets import MNIST
MNIST_ROOT = "./MNIST"

#### -- PARAMETERS -- ####
image_size = 28
window_size = 14
image_numbers = 5
saccades_on_each_image = 5
iterations = 1000
layer_5_6_size = 24
higher_layer_size = 16
# pos_x, pos_y = generate_walk(length=100, R=10)
screen_shot_path = "C:\\Users\\amilion\\Desktop\\develop\\python\\NS\\records\\L5.6"

time_window = 50
crop_iteration = 2
Input_Width = 28
Input_Height = 28
Crop_Window_Width = 21
Crop_Window_Height = 21
DoG_SIZE = 5
rest_interval = 20

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

OUT_CHANNEL = 8
IN_CHANNEL = 1
KERNEL_WIDTH = 13
KERNEL_HEIGHT = 13

# INPUT_WIDTH = IMAGE_WIDTH - DoG_SIZE + 1
# INPUT_HEIGHT = IMAGE_HEIGHT - DoG_SIZE + 1
INPUT_WIDTH = Crop_Window_Width - DoG_SIZE + 1
INPUT_HEIGHT = Crop_Window_Height - DoG_SIZE + 1

L4_WIDTH = INPUT_WIDTH - KERNEL_WIDTH + 1
L4_HEIGHT = INPUT_HEIGHT - KERNEL_HEIGHT + 1

L23_WIDTH = L4_WIDTH//2
L23_HEIGHT = L4_HEIGHT//2

J_0 = 300
p = 0.8

dataset_directory_path = "./first_step"

transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels = 1), # not necessary
    Conv2dFilter(DoGFilter(size = 5, sigma_1 = 4, sigma_2 = 1, zero_mean=True, one_sum=True).unsqueeze(0).unsqueeze(0)),
    SqueezeTransform(dim = 0),
    SimplePoisson(time_window = time_window , ratio = 2),
])


dataset = MNIST(root=MNIST_ROOT, train=True, download=False, transform=transformation)
first_class = dataset.data[dataset.targets == 4][:70]
second_class = dataset.data[dataset.targets == 9][:70]

target = [0] * len(first_class) + [1] * len(second_class)
target = torch.Tensor(target)

two_class_dataset = torch.cat((first_class, second_class), dim=0)
new_dataset_size = first_class.shape[0] + second_class.shape[0]

t = torch.arange(new_dataset_size)
np.random.shuffle(t.numpy())
two_class_dataset = two_class_dataset[t]
target = target[t]

target_new = [[i] * time_window * crop_iteration for i in target]
target_new = torch.Tensor(target_new)
target = target_new.view(-1)

new_dataset = torch.empty(0,Crop_Window_Width - DoG_SIZE + 1, Crop_Window_Height - DoG_SIZE + 1)
centers = []


def load_image(path,size = None):
    img = cv2.imread(path)
    if(size):
        img = cv2.resize(img,size)
    return torch.tensor(img[:,:,0],dtype=torch.float32)

def show_image(image,normal=False):
    plt.axis("off")
    if(normal):
        plt.imshow(image,cmap='gray',vmin=0,vmax=255)
    else:
        plt.imshow(image,cmap='gray')
    plt.show()
    

def show_filters(weight):
    fig,axes = plt.subplots(1,weight.shape[0])
    fig.set_size_inches(5*weight.shape[0], 5)
    # fig.suptitle(f'plots of synaptic share weights for d = {weight.shape[0]}')
    for i in range(weight.shape[0]):
        axes[i].imshow(weight[i][0],cmap='gray')
        axes[i].axis('off')
        
        
def show_images(imgs,title,count):
    fig,axes = plt.subplots(1,count)
    fig.set_size_inches(5*count, 5)
    plt.text(x=0.5, y=0.94, s=title, fontsize=28, ha="center", transform=fig.transFigure)
    for i in range(count):
        axes[i].imshow(imgs[i][0][0],cmap='gray')
        axes[i].axis('off')


def confidence_crop_interspace(inp_width, inp_height, window_width, window_height):
    x1 = window_width//2
    x2 = (inp_width - 1) - (window_width//2)
    y1 = window_height//2 
    y2 = (inp_height - 1) - (window_height//2)

    center_x = random.randint(x1, x2)
    center_y = random.randint(y1, y2)
    center_coordinates = [center_x, center_y]
    top_left_x = center_x - (window_width//2)
    top_left_y = center_y - (window_height//2)
    top_left_coordinates = [top_left_x, top_left_y]
    coordinates = [center_coordinates, top_left_coordinates]

    return coordinates


dl = DataLoader(two_class_dataset,shuffle=False)

#### -- NETWORK INTIALIZING -- ####

net = Network(behavior=prioritize_behaviors([TimeResolution(dt=1)]))

#### -- NEURON GROUP INTIALIZING -- ####
layer_5_6 = NeuronGroup(
    net=net,
    size=NeuronDimension(width=layer_5_6_size, height=layer_5_6_size),
    behavior=prioritize_behaviors(
        [
            SimpleDendriteStructure(),
            SimpleDendriteComputation(apical_provocativeness=0.9),
            Fire(),
            KWTA(k=10),
            NeuronAxon(),
        ]
    )
    | (
        {
            250: ConstantCurrent(scale=4),
            260: GPCell(
                R=8,
                tau=5,
                threshold=-30,
                v_rest=-65,
                v_reset=-67,
                L=15,
                I_amp = 20,
                # V=speed_vector_converter(pos_x, pos_y),
                init_v=torch.tensor([-67]).expand(layer_5_6_size * layer_5_6_size).clone().to(dtype=torch.float32)
            ),
            600: Recorder(["I", "v"]),
            601: EventRecorder(["spikes"]),
        }
    ),
)

loader_neuron_group = NeuronGroup(
    net=net,
    size=NeuronDimension(depth=1, height=window_size, width=window_size),
    behavior=prioritize_behaviors(
        [
            SimpleDendriteStructure(),
            SimpleDendriteComputation(),
            LIF(
                R=10,
                tau=5, 
                v_reset=-67,
                v_rest=-67,
                threshold=-60,
            ),
            Fire(),
            NeuronAxon(),
        ]) | {
            270: OnlineDataLoader(
                # data_set=torch.rand(image_numbers, image_size, image_size), 
                data_set=dl.dataset,
                batch_number=saccades_on_each_image,
                rest_interval=rest_interval,
                iterations=iterations
            ),
            600: Recorder(["focus_loc"]),
            601: EventRecorder(["spikes"]),
        }
)



higher_layer = NeuronGroup(
    net=net,
    size=NeuronDimension(depth=1, height=higher_layer_size, width=higher_layer_size),
    behavior=prioritize_behaviors(
        [
            SimpleDendriteStructure(),
            SimpleDendriteComputation(),
            NeuronAxon(),
        ]
    )
    | (
        {
            250: RandomInputCurrent(prob_to_spike=0.1, T=10),
            # 600: Recorder(["I", "v"]),
            603: EventRecorder(["spikes"]),
        }
    ),
)

#### -- SYNAPSE GROUP INTIALIZING -- ####


sg = SynapseGroup(
    net=net,
    src=higher_layer,
    dst=layer_5_6,
    tag="Apical, exi",
    behavior=prioritize_behaviors(
        [SynapseInit(), SimpleDendriticInput() ,WeightInitializer(mode="normal(0.2, 3)")]
    )
    # | (
    #     {
    #         180: GPCellLateralInhibition(max_inhibition=3, r=3, n=9),
    #     }
    # ),
)

Loader_to_GP = SynapseGroup(
    net=net,
    src=loader_neuron_group,
    dst=layer_5_6,
    behavior={
        275: LocationCoder(),
    }
)

GP_lateral = SynapseGroup(
    net=net,
    src=layer_5_6,
    dst=layer_5_6,
    tag="Proximal",
    behavior=prioritize_behaviors(
        [SynapseInit(), SimpleDendriticInput()]
    )
    | (
        {
            180: GPCellLateralInhibition(kernel_side=29, max_inhibition=2, r=13, n=2, inhibitory=1, near_by_excitatory=3),
        }
    ),
)




#### -- RUNNING THE MODEL -- ####

net.initialize()
net.simulate_iterations(iterations)

""" 
#######################################################
##################   Visualization   ##################
#######################################################
"""

# print(layer_5_6["I"])

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = list(prop_cycle.by_key()["color"])


ngs = [layer_5_6]
for i in range(0, 100):
    cnt = 0
    for ng in ngs:
        iter_spike_multi_real(
            loader_neuron_group["focus_loc"][0],
            ng,
            itr=i,
            step=1,
            color=colors[cnt],
            save=True,
            lib=screen_shot_path,
            label="GPCell" + str(cnt + 1),
            offset_x=0,
            offset_y=0,
            base_offset_x=0,
            base_offset_y=0,
        )
        cnt += 1
        # break

    plt.clf()

