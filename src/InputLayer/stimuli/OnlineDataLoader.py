from typing import Tuple
import random


import torch
from torchvision.transforms.functional import crop
import pymonntorch as pynt

from conex import *

def hypo_func(
    image: torch.Tensor,      
) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = random.randint(1, 10), random.randint(1, 10)
    return (torch.tensor([x, y]), torch.rand(28, 28))

# def confidence_crop_interspace(inp_width, inp_height, window_width, window_height):
def confidence_crop_interspace(image: torch.Tensor, window_width: int, window_height: int, device="cpu"):
    inp_width = image.size(0)
    inp_height = image.size(1) 
    x1 = window_width//2
    x2 = (inp_width - 1) - (window_width//2)
    y1 = window_height//2 
    y2 = (inp_height - 1) - (window_height//2)

    """
    5 fixed points on the image to saccade
    """
    cent_x = [window_width//2, window_width//2, inp_width - window_width//2, inp_width - window_width//2, inp_width//2]
    cent_y = [window_height//2, inp_height - window_height//2, window_height//2, inp_height - window_height//2, inp_height//2]

    opt = random.randint(0, 4)
    center_x = cent_x[opt]
    center_y = cent_y[opt]
    
    # import pdb;pdb.set_trace() 
    
    # center_x = random.randint(x1, x2)
    # center_y = random.randint(y1, y2)
    center_coordinates = [center_x, center_y]
    top_left_x = center_x - (window_width//2)
    top_left_y = center_y - (window_height//2)
    top_left_coordinates = [top_left_x, top_left_y]
    coordinates = torch.tensor([center_coordinates, top_left_coordinates]).to(device)

    return (coordinates, crop(img=image, top=top_left_y, left=top_left_x, height = window_height, width = window_width))

class OnlineDataLoader(pynt.Behavior):
    """
        parameter:
            (Tensor) data_set: the data set to use for network training.
            (int) batch_number: number of batch to be cropped from each data set image. 
            (float) ratio: A scale factor for probability of spiking.
            (int) iterations: the number of simulation iterations.
    """
    def __init__(self, 
        data_set: torch.Tensor,
        targets: torch.Tensor,
        saccade_iterations: int,
        train_images_number: int,
        test_images_number: int,
        train_iterations: int,
        rest_iterations: int,
        test_iterations: int,
        window_size: int,
        max_image_iterations : int,
        ratio: float = 1,
        inter_image_interval: int = 5,
        *args, 
        **kwargs,
    ):
        super().__init__(
            data_set = data_set, 
            targets = targets,
            saccade_iterations = saccade_iterations, 
            train_iterations=train_iterations, 
            ratio=ratio, 
            window_size=window_size,
            rest_iterations=rest_iterations, 
            train_images_number = train_images_number, 
            test_images_number = test_images_number,
            inter_image_interval = inter_image_interval,
            test_iterations = test_iterations,
            max_image_iterations = max_image_iterations,
            *args, **kwargs)

    def initialize(self, neuron):
        neuron.network.image_idx = 0
        neuron.network.iter_counter = 0

        self.train_images_number = self.parameter("train_images_number", required=True)
        self.test_images_number = self.parameter("test_images_number", required=True)

        self.rest_iterations = self.parameter("rest_iterations", required=True)
        self.test_iterations = self.parameter("test_iterations", required=True)
        self.train_iterations = self.parameter("train_iterations", required=True)

        self.max_image_iterations = self.parameter("max_image_iterations", required=True)
        
        self.data_set = self.parameter("data_set", required=True)
        self.targets = self.parameter("targets", required=True)

        self.saccade_iterations = self.parameter("saccade_iterations", required=True)
        self.ratio = self.parameter("ratio", 2)

        self.inter_image_interval = self.parameter("inter_image_interval", 5)
        self.window_size = self.parameter("window_size", required=True)
        self.poisson_coder = SimplePoisson(time_window=1, ratio=self.ratio)

        self.saccade_infos = confidence_crop_interspace(self.data_set[neuron.network.image_idx], window_height=self.window_size, window_width=self.window_size)
        neuron.focus_loc = self.saccade_infos[0][0]
        return super().initialize(neuron)

    def forward(self, neuron):
        # rest phase between train & test
        if(neuron.network.phase == "rest") :
            return super().forward(neuron) 
        
        # end of train
        if(neuron.network.phase == "train" and neuron.network.image_idx >= self.train_images_number) :
            return super().forward(neuron) 

        # end of test
        if(neuron.network.phase == "test" and neuron.network.image_idx >= self.test_images_number) :
            return super().forward(neuron) 
        
        # inter image rest interval
        if(neuron.network.iter_counter <= self.inter_image_interval) : 
            neuron.network.iter_counter += 1
            neuron.focus_loc = torch.tensor([-1, -1]).to(neuron.device)
            return super().forward(neuron) 

        # next 
        if(neuron.network.iter_counter > self.max_image_iterations) : 
            neuron.network.image_idx += 1
            neuron.network.iter_counter = 0

        # saccade
        if(neuron.network.iter_counter % self.saccade_iterations == 0) : 
            self.saccade_infos = confidence_crop_interspace(self.data_set[neuron.network.image_idx], window_height=self.window_size, window_width=self.window_size)
            neuron.focus_loc = self.saccade_infos[0][0]

        spikes = self.poisson_coder(img=self.saccade_infos[1])
        neuron.network.targets = neuron.network.network_target[neuron.network.image_idx]
        neuron.v[spikes.view(-1)] = neuron.threshold + 1e-2
        neuron.network.iter_counter += 1

        return super().forward(neuron)
    

