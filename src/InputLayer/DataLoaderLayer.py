from pymonntorch import *
from conex import *

from src.InputLayer.stimuli.OnlineDataLoader import OnlineDataLoader

class DataLoaderLayer():
    def __init__(
        self,
        net,
        data_loader,
        widnow_size,
        saccades_on_each_image,
        iterations,
    ):
        self.net = net
        self.dl = data_loader
        self.window_size = widnow_size
        self.saccades_on_each_image = saccades_on_each_image
        self.iterations = iterations
        super().__init__(self)
        
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
                NeuronAxon(),
            ]) | {
                270: OnlineDataLoader(
                    # data_set=self.dl.dataset, 
                    data_set=torch.rand(5, widnow_size, widnow_size,),
                    batch_number=self.saccades_on_each_image,
                    iterations=self.iterations
                ),
                600: Recorder(["focus_loc"]),
                601: EventRecorder(["spikes"]),
            }
        )
        return Layer(
            net=self.net,
            neurongroups = [loader_neuron_group],
            tag="loader_layer"
)

