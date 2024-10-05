from typing import Tuple
import random


import torch
import pymonntorch as pynt

from conex import *

def hypo_func(
    image: torch.Tensor,      
) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = random.randint(1, 10), random.randint(1, 10)
    return (torch.tensor([x, y]), torch.rand(28, 28))

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
        iterations: int,
        ratio: float = 1,
        *args, 
        **kwargs,
    ):
        super().__init__(data_set = data_set, batch_number = batch_number, iterations=iterations, ratio=ratio, *args, **kwargs)

    def initialize(self, neuron):
        self.data_set = self.parameter("data_set", required=True)
        self.batch_number = self.parameter("batch_number", required=True)
        self.ratio = self.parameter("ratio", 1)
        self.iterations = self.parameter("iterations", required=True)
        self.poisson_coder = SimplePoisson(time_window=1, ratio=self.ratio)
        self.saccade_infos = hypo_func(self.data_set[0])
        self.interval = self.iterations // self.data_set.size(0)
        neuron.focus_loc = self.saccade_infos[0]
        return super().initialize(neuron)

    def forward(self, neuron):
        image_idx =  neuron.network.iteration // self.interval
        if image_idx < self.data_set.size(0) and neuron.network.iteration % (self.interval // self.batch_number) == 0:
            self.saccade_infos = hypo_func(self.data_set[image_idx])
        neuron.focus_loc = self.saccade_infos[0]
        spikes = self.poisson_coder(img=self.saccade_infos[1])
        neuron.v[spikes.view(-1)] = neuron.threshold + 1e-2
        return super().forward(neuron)
    

