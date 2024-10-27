from pymonntorch import *
from conex import *

from InputLayer.stimuli.OnlineDataLoader import OnlineDataLoader

class DataLoaderLayer():
    def __init__(
        self,
        net,
        train_data_loader,
        test_data_loader,
        widnow_size,
        saccades_on_each_image,
        rest_interval,
        train_iterations,
        test_iterations,
        phase_iterations = 20,
    ):
        self.net = net
        self.train_dl = train_data_loader
        self.test_dl = test_data_loader,
        self.window_size = widnow_size
        self.saccades_on_each_image = saccades_on_each_image
        self.train_iterations = train_iterations
        self.rest_interval = rest_interval
        self.test_iterations = test_iterations
        self.phase_iterations = phase_iterations
        
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
                    train_data_set=self.train_dl.dataset, 
                    test_data_set = self.test_dl.dataset,
                    window_size=self.window_size,
                    batch_number=self.saccades_on_each_image,
                    train_iterations=self.train_iterations,
                    rest_interval = self.rest_interval,
                    phase_interval= self.phase_iterations,
                    test_iterations= self.test_iterations
                ),
                600: Recorder(["focus_loc"]),
                601: EventRecorder(["spikes"]),
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

