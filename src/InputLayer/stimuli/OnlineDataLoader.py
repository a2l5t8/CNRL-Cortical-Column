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

def middle_crop_interspace(image: torch.Tensor, window_width: int, window_height: int):
    x = (image.size(0) // (2 * window_width))
    y = (image.size(1) // (2 * window_height))
    
    center_coordinates = [x + window_width // 2, y + window_height //2]
    top_left_coordinates = [x, y]
    
    coordinates = torch.Tensor([center_coordinates, top_left_coordinates])
    
    return (coordinates, crop(img=image, top=y, left=x, height=window_height, width=window_width))


class OnlineDataLoader(pynt.Behavior):
    """
        parameter:
            (Tensor) data_set: the data set to use for network training.
            (int) batch_number: number of batch to be cropped from each data set image. 
            (float) ratio: A scale factor for probability of spiking.
            (int) iterations: the number of simulation iterations.
    """
    def __init__(self, 
        train_data_set: torch.Tensor,
        test_data_set: torch.Tensor,
        batch_number: int,
        train_iterations: int,
        test_iterations: int,
        window_size: int,
        ratio: float = 1,
        rest_interval: int = 5,
        phase_interval: int = 20,
        *args, 
        **kwargs,
    ):
        super().__init__(
            train_data_set = train_data_set, 
            test_data_set = test_data_set,
            batch_number = batch_number, 
            train_iterations=train_iterations, 
            test_iterations=test_iterations,
            ratio=ratio, 
            window_size=window_size,
            rest_interval=rest_interval, 
            phase_interval=phase_interval,
            *args, **kwargs)

    def initialize(self, neuron):
        self.train_data_set = self.parameter("train_data_set", required=True)
        self.test_data_set = self.parameter("test_data_set", required=True)
        self.batch_number = self.parameter("batch_number", required=True)
        self.phase_interval = self.parameter("phase_interval", 20)
        self.ratio = self.parameter("ratio", 2)
        self.train_iterations = self.parameter("train_iterations", required=True)
        self.test_iterations = self.parameter("test_iterations", required=True)
        self.rest_interval = self.parameter("rest_interval", 5)
        self.rest_for_test = 10
        self.window_size = self.parameter("window_size", required=True)
        self.poisson_coder = SimplePoisson(time_window=1, ratio=self.ratio)
        
        ### for train phase
        
        self.saccade_infos = confidence_crop_interspace(self.train_data_set[0], window_height=self.window_size, window_width=self.window_size)
        self.train_interval = self.train_iterations // self.train_data_set.size(0)
        neuron.focus_loc = self.saccade_infos[0][0]
        
        # TO DO -> test_dataset_shape must match train_data_set_shape ## Error Handling
        
        ### for test phase
        self.test_interval = self.test_iterations // self.test_data_set.size(0)
        
        return super().initialize(neuron)

    def forward(self, neuron):
        ### Train run
        if (neuron.network.iteration <= self.train_iterations):
            image_idx =  neuron.network.iteration // self.train_interval
            if(image_idx >= self.train_data_set.size(0)): 
                return super().forward(neuron)
            neuron.network.targets = neuron.network.network_target[image_idx]
            if neuron.network.iteration % self.train_interval >= self.train_interval - self.rest_interval:
                neuron.focus_loc = torch.tensor([-1, -1])
                return super().forward(neuron)
            if image_idx < self.train_data_set.size(0) and neuron.network.iteration % ((self.train_interval - self.rest_interval) // self.batch_number) == 0:
                self.saccade_infos = confidence_crop_interspace(self.train_data_set[image_idx], window_height=self.window_size, window_width=self.window_size)
            neuron.focus_loc = self.saccade_infos[0][0]
            if self.train_interval - self.rest_interval > neuron.network.iteration - image_idx * self.train_interval:
                spikes = self.poisson_coder(img=self.saccade_infos[1])
                neuron.v[spikes.view(-1)] = neuron.threshold + 1e-2
            return super().forward(neuron)
            
        ### Rest between phase
        if (neuron.network.iteration <= self.train_iterations + self.phase_interval):
            neuron.focus_loc = torch.tensor([-1, -1])
            return super().forward(neuron)

        itr = neuron.network.iteration - self.train_iterations - self.phase_interval
        rest_for_test = 10
        ### Test run
        image_idx = itr // self.test_interval
        saccade_infos = middle_crop_interspace(image = self.test_data_set[image_idx], window_width=self.window_size, window_height=self.window_size)
        neuron.focu_loc = saccade_infos[0][0]
        if (image_idx >= self.test_data_set.size(0) or self.test_interval - self.rest_for_test < itr % (image_idx * self.test_interval)):
            return super().forward(neuron)
        
        spikes = self.poisson_coder(img=saccade_infos[1])
        neuron.v[spikes.view(-1)] = neuron.threshold + 1e-2
        
        
        
        
        

