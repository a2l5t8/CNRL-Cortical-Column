from pymonntorch import *
from conex import *

from InputLayer.stimuli.OnlineDataLoader import OnlineDataLoader

class DataLoaderLayer():
    def __init__(
        self,
        net,
        data_loader,
        targets,
        widnow_size,
        saccades_on_each_image,
        rest_interval,
        train_iterations,
        phase_interval,
        train_images_number,
        test_images_number ,
        test_iterations
    ):
        self.net = net
        self.dl = data_loader
        self.targets = targets
        self.window_size = widnow_size
        self.saccades_on_each_image = saccades_on_each_image
        self.rest_interval = rest_interval
        self.train_iterations = train_iterations
        self.phase_interval = phase_interval
        self.train_images_number = train_images_number
        self.test_images_number = test_images_number
        self.test_iterations = test_iterations
        
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
                    batch_number=self.saccades_on_each_image,
                    train_iterations=self.train_iterations,
                    rest_interval = self.rest_interval,
                    phase_interval=self.phase_interval,
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

