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
def confidence_crop_interspace(image: torch.Tensor, window_width: int, window_height: int):
    inp_width = image.size(0)
    inp_height = image.size(1) 
    x1 = window_width//2
    x2 = (inp_width - 1) - (window_width//2)
    y1 = window_height//2 
    y2 = (inp_height - 1) - (window_height//2)
    
    # import pdb;pdb.set_trace() 
    
    center_x = random.randint(x1, x2)
    center_y = random.randint(y1, y2)
    center_coordinates = [center_x, center_y]
    top_left_x = center_x - (window_width//2)
    top_left_y = center_y - (window_height//2)
    top_left_coordinates = [top_left_x, top_left_y]
    coordinates = torch.tensor([center_coordinates, top_left_coordinates])

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
        batch_number: int,
        train_images_number: int,
        test_images_number: int,
        train_iterations: int,
        phase_interval: int,
        test_iterations: int,
        window_size: int,
        ratio: float = 1,
        rest_interval: int = 5,
        *args, 
        **kwargs,
    ):
        super().__init__(
            data_set = data_set, 
            batch_number = batch_number, 
            train_iterations=train_iterations, 
            ratio=ratio, 
            window_size=window_size,
            rest_interval=rest_interval, 
            train_images_number = train_images_number, 
            test_images_number = test_images_number,
            phase_interval = phase_interval,
            test_iterations = test_iterations,
            *args, **kwargs)

    def initialize(self, neuron):
        self.train_images_number = self.parameter("train_images_number", required=True)
        self.test_images_number = self.parameter("test_images_number", required=True)
        self.phase_interval = self.parameter("phase_interval", required=True)
        self.test_iterations = self.parameter("test_iterations", required=True)
        self.data_set = self.parameter("data_set", required=True)
        self.batch_number = self.parameter("batch_number", required=True)
        self.ratio = self.parameter("ratio", 2)
        self.train_iterations = self.parameter("train_iterations", required=True)
        self.rest_interval = self.parameter("rest_interval", 5)
        self.window_size = self.parameter("window_size", required=True)
        self.poisson_coder = SimplePoisson(time_window=1, ratio=self.ratio)
        self.saccade_infos = confidence_crop_interspace(self.data_set[0], window_height=self.window_size, window_width=self.window_size)
        self.train_interval = self.train_iterations // self.train_images_number
        self.test_interval = self.test_iterations // self.test_images_number
        neuron.focus_loc = self.saccade_infos[0][0]
        return super().initialize(neuron)

    def forward(self, neuron):
        if 0 < neuron.network.iteration - self.train_iterations and neuron.network.iteration - self.train_iterations <= self.phase_interval:
            return super().forward(neuron) 
        if neuron.network.iteration  > self.train_iterations + self.phase_interval:
            itr = neuron.network.iteration - self.train_iterations - self.phase_interval
            image_idx = itr // self.test_interval + self.train_images_number  
            if(image_idx >= self.data_set.size(0)): 
                return super().forward(neuron)
            if itr % self.test_interval >= self.test_interval - self.rest_interval:
                neuron.focus_loc = torch.tensor([-1, -1])
                return super().forward(neuron)
            if image_idx < self.data_set.size(0) and itr % ((self.test_interval - self.rest_interval) // self.batch_number) == 0:
                self.saccade_infos = confidence_crop_interspace(self.data_set[image_idx], window_height=self.window_size, window_width=self.window_size)
            neuron.focus_loc = self.saccade_infos[0][0]
            if self.test_interval - self.rest_interval > itr % self.test_interval:
                spikes = self.poisson_coder(img=self.saccade_infos[1])
                neuron.v[spikes.view(-1)] = neuron.threshold + 1e-2
            return super().forward(neuron)
        image_idx =  neuron.network.iteration // self.train_interval
        neuron.network.targets = neuron.network.network_target[image_idx]
        if neuron.network.iteration % self.train_interval >= self.train_interval - self.rest_interval:
            neuron.focus_loc = torch.tensor([-1, -1])
            return super().forward(neuron)
        if image_idx < self.data_set.size(0) and neuron.network.iteration % ((self.train_interval - self.rest_interval) // self.batch_number) == 0:
            self.saccade_infos = confidence_crop_interspace(self.data_set[image_idx], window_height=self.window_size, window_width=self.window_size)
        neuron.focus_loc = self.saccade_infos[0][0]
        if self.train_interval - self.rest_interval > neuron.network.iteration - image_idx * self.train_interval:
            spikes = self.poisson_coder(img=self.saccade_infos[1])
            neuron.v[spikes.view(-1)] = neuron.threshold + 1e-2
        return super().forward(neuron)
    

