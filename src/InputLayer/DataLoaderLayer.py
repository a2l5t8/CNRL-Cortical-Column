from pymonntorch import *
from conex import *

from InputLayer.stimuli.OnlineDataLoader import OnlineDataLoader

class DataLoaderLayer():
    def __init__(
        self,
        net,
        data_loader,
        targets,
        window_size,
        saccade_iterations,
        inter_image_interval,
        train_iterations,
        rest_iterations,
        train_images_number,
        test_images_number ,
        test_iterations,
        max_image_iterations
    ):
        self.net = net
        self.dl = data_loader
        self.targets = targets
        self.window_size = window_size
        self.saccade_iterations = saccade_iterations
        self.inter_image_interval = inter_image_interval
        self.train_iterations = train_iterations
        self.rest_iterations = rest_iterations
        self.train_images_number = train_images_number
        self.test_images_number = test_images_number
        self.test_iterations = test_iterations
        self.max_image_iterations = max_image_iterations
        
    def build_data_loader(self):
        loader_neuron_group = NeuronGroup(
            net=self.net,
            size=NeuronDimension(depth=1, height=self.window_size, width=self.window_size),
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
                    SpikeTrace(tau_s = 3),
                    NeuronAxon(),
                ]) | {
                    270: OnlineDataLoader(
                        data_set=self.dl.dataset, 
                        targets=self.targets,
                        window_size=self.window_size,
                        saccade_iterations=self.saccade_iterations,
                        train_iterations=self.train_iterations,
                        inter_image_interval = self.inter_image_interval,
                        rest_iterations=self.rest_iterations,
                        max_image_iterations = self.max_image_iterations,
                        train_images_number=self.train_images_number,
                        test_images_number=self.test_images_number,
                        test_iterations=self.test_iterations
                    ),
                }
        )
        return Layer(
            net=self.net,
            neurongroups = [loader_neuron_group],
            tag="loader_layer",
            output_ports={
                "data_out": (
                    None,
                    [Port(object = loader_neuron_group, label = None)]
                )
            }
        )

