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


#######################################################
################ Prioritize Behaviors #################
#######################################################


prioritize_behaviors([
    
    LateralDendriticInput(current_coef=60, inhibitory=True),
    Conv2dDendriticInput(current_coef = 1, stride = 1, padding = 0),
    AveragePool2D(current_coef = 1),
    SimpleDendriteStructure(),
    SimpleDendriteComputation(),
    LIF(
        tau = 10,
        R = 1,
        threshold = -13,
        v_rest = -65,
        v_reset = -70
    ),
    InherentNoise(scale = 1, offset = 0),
    SpikeNdDataset(
        dataloader = "sensory",
        instance_duration = 1
    ),
    Fire(),
    NeuronAxon(),
])


#######################################################
#################### Functions ########################
#######################################################


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


#######################################################
####################### Dataset #######################
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
show_image(torch.sum(dataset[0][0],0))


#######################################################
######################## Config #######################
#######################################################


DoG_SIZE = 5
IMAGE_WIDTH = 14
IMAGE_HEIGHT = 14

OUT_CHANNEL = 5
IN_CHANNEL = 1
KERNEL_WIDTH = 10
KERNEL_HEIGHT = 10

INPUT_WIDTH = IMAGE_WIDTH - DoG_SIZE + 1
INPUT_HEIGHT = IMAGE_HEIGHT - DoG_SIZE + 1

L4_WIDTH = INPUT_WIDTH - KERNEL_WIDTH + 1
L4_HEIGHT = INPUT_HEIGHT - KERNEL_HEIGHT + 1

L23_WIDTH = L4_WIDTH//2
L23_HEIGHT = L4_HEIGHT//2

J_0 = 300
p = 0.8


#######################################################
####################### Network #######################
#######################################################



net = Neocortex(dt=1, dtype=torch.float32)



#################### Input Layer #####################



input_layer = InputLayer(
    net=net,
    input_dataloader= dl,
    sensory_data_dim=2,
    sensory_size = NeuronDimension(depth=1, height = INPUT_HEIGHT, width = INPUT_WIDTH),
    sensory_trace= 3,
    instance_duration= time_window,
    silent_interval=100,
)




#################### L4 ####################



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
    KWTA(k=5),
    ActivityBaseHomeostasis(window_size=10, activity_rate=200, updating_rate=0.0001),
    Fire(),
    SpikeTrace(tau_s = 15),
    NeuronAxon(),
]) | ({
    800 : Recorder(variables = ['v', "I", "torch.mean(I)", "trace", "n.spikes.sum()/n.size"]),
    801 : EventRecorder(['spikes'])
}))

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
]) | ({
    800 : Recorder(variables = ['v', "I", "torch.mean(I)", "trace", "n.spikes.sum()/n.size"]),
    801 : EventRecorder(['spikes'])
}))


sgi4e = SynapseGroup(net = net, src = input_layer.sensory_pop, dst = ng4e, tag = "Proximal", behavior = prioritize_behaviors([
    SynapseInit(),
    WeightInitializer(weights = torch.normal(0.5, 2, (OUT_CHANNEL, IN_CHANNEL, KERNEL_HEIGHT, KERNEL_WIDTH)) ),
    Conv2dDendriticInput(current_coef = 35 , stride = 1, padding = 0),
    Conv2dSTDP(a_plus=0.05, a_minus=0.004),
    WeightNormalization(norm = 50)
]))


sg4e4i = SynapseGroup(net = net, src = ng4e, dst = ng4i, tag = "Proximal", behavior = prioritize_behaviors([
    SimpleDendriticInput(),
    SynapseInit(),
    WeightInitializer(mode = "ones", scale = J_0/math.sqrt(400 * p), density = 0.02, true_sparsity = False),
]))


sg4i4e = SynapseGroup(net = net, src = ng4i, dst = ng4e, tag = "Proximal", behavior = prioritize_behaviors([
    SimpleDendriticInput(),
    SynapseInit(),
    WeightInitializer(mode = "ones", scale = J_0/math.sqrt(400 * p), density = 0.02, true_sparsity = False),
]))

sg4e4e = SynapseGroup(net = net, src = ng4e, dst = ng4e, tag = "Proximal", behavior=prioritize_behaviors([
    SynapseInit(),
    WeightInitializer(weights=torch.Tensor([1, 1, 1, 1, 0, 1, 1, 1, 1]).view(1, 1, 9, 1, 1)),
    LateralDendriticInput(current_coef=1300, inhibitory = True),
])| {
    600 : Recorder(["I"])
})

sg4i4i = SynapseGroup(net = net, src = ng4i, dst = ng4i, tag = "Proximal", behavior = prioritize_behaviors([
    SimpleDendriticInput(),
    SynapseInit(),
    WeightInitializer(mode = "ones", scale = J_0/math.sqrt(400 * p), density = 0.02, true_sparsity = False),
]))



#################### L2&3 ####################



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
]) | ({
    800 : Recorder(['v', "I", "torch.mean(I)", "trace", "n.spikes.sum()/n.size"]),
    801 : EventRecorder(['spikes'])
}))

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
]) | ({
    800 : Recorder(['v', "I", "torch.mean(I)", "trace", "n.spikes.sum()/n.size"]),
    801 : EventRecorder(['spikes'])
}))

sg4e23e = SynapseGroup(net = net, src = ng4e, dst = ng23e, tag = "Proximal", behavior = prioritize_behaviors([
    SynapseInit(),
    # WeightInitializer(mode = "normal(4, 5)", density = 0.02, true_sparsity = False),
    AveragePool2D(current_coef = 150),
]) | ({
    800 : Recorder(["I"]),
}))

sg23e23i = SynapseGroup(net = net, src = ng23e, dst = ng23i, tag = "Proximal", behavior = prioritize_behaviors([
    SimpleDendriticInput(),
    SynapseInit(),
    WeightInitializer(mode = "ones", scale = J_0/math.sqrt(625 * p), density = 0.02, true_sparsity = False),
]) | ({
    800 : Recorder(["I"]),
}))

sg23i23e = SynapseGroup(net = net, src = ng23i, dst = ng23e, tag = "Proximal", behavior = prioritize_behaviors([
    SimpleDendriticInput(),
    SynapseInit(),
    WeightInitializer(mode = "ones", scale = J_0/math.sqrt(625 * p), density = 0.02, true_sparsity = False),
]) | ({
    800 : Recorder(["I"]),
}))

sg23i23i = SynapseGroup(net = net, src = ng23i, dst = ng23i, tag = "Proximal", behavior = prioritize_behaviors([
    SimpleDendriticInput(),
    SynapseInit(),
    WeightInitializer(mode = "ones", scale = J_0/math.sqrt(625 * p), density = 0.02, true_sparsity = False),
]) | ({
    800 : Recorder(["I"]),
}))


#######################################################
#################### Visualization ####################
#######################################################


net.initialize()
net.simulate_iterations(3000)

show_filters(sgi4e.weights)
