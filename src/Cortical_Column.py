#######################################################
######################## Setup ########################
#######################################################

from conex import *
from pymonntorch import *

from matplotlib import pyplot as plt
from PIL import Image

import torchvision
from torch.utils.data import DataLoader

from conex.helpers.filters import DoGFilter

from L423.tools.visualize import *
from L56.RefrenceFrames import RefrenceFrame
from L56.stimuli.current_base import RandomInputCurrent
from L56.synapse.vDistributor import ManualVCoder
from InputLayer.DataLoaderLayer import DataLoaderLayer
from InputLayer.synapse.LocationCoder import LocationCoder
from L423.network.SetTarget import SetTarget
from L423.L423 import L4, L23
from FC import fullyConnected
from FC.synapse.learning import AttentionBasedRSTDP
from FC.tools.model_evaluation import accuracy_score

from L56.tools.visualization import refrence_frame_raster
from L423.tools.visualize import show_filters

from torchvision.datasets import MNIST

#######################################################
######################## Config #######################
#######################################################


REFRENCE_FRAME_DIM = 23
REFRENCE_INH_DIM = 15
SCREEN_SHOT_PATH = "C:\\Users\\amilion\\Desktop\\develop\\python\\NS\\records\\L5.6"

Input_Width = 28
Input_Height = 28
Crop_Window_Width = 23
Crop_Window_Height = 23
DoG_SIZE = 5

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_NUMBER = 100

OUT_CHANNEL = 16
IN_CHANNEL = 1
KERNEL_WIDTH = 13
KERNEL_HEIGHT = 13

DATASET_IMAGE_WIDTH = IMAGE_WIDTH - DoG_SIZE + 1
DATASET_IMAGE_HEIGHT = IMAGE_HEIGHT - DoG_SIZE + 1
INPUT_WIDTH = Crop_Window_Width - DoG_SIZE + 1
INPUT_HEIGHT = Crop_Window_Height - DoG_SIZE + 1

L4_WIDTH = Crop_Window_Width - KERNEL_WIDTH + 1
L4_HEIGHT = Crop_Window_Height - KERNEL_HEIGHT + 1

L23_WIDTH = L4_WIDTH//2
L23_HEIGHT = L4_HEIGHT//2

J_0 = 300
p = 0.8

TRAIN_IMAGES_NUMBER = 70
TEST_IMAGES_NUMBER = 40
TRAIN_ITERATIONS = 14000
REST_INTERVAL = 100
TEST_ITERATIONS = 8000

SACCADE = 3


class NeoCorticalColumn():
    """
        Note: Do not use this container, not completed yet!
    """
    def __init__(
        self,
        net: Neocortex = None,
    ):  
        
        self.net = net
        if not self.net:
            self.net = net = Neocortex(
                    dt = 1, 
                    dtype = torch.float32, 
                    behavior = {
                            5 : SetTarget(target = target), 
                            601 : Recorder(["dopamine"]),
                        },
                    index = True
                )

        ### layers
        
        self.L56 = RefrenceFrame(
            net=net, 
            k=5, 
            refrence_frame_side=REFRENCE_FRAME_DIM, 
            inhibitory_size=REFRENCE_INH_DIM,
            competize=True
        )

        self.L4 = L4(
            net = net, 
            IN_CHANNEL = IN_CHANNEL, 
            OUT_CHANNEL = OUT_CHANNEL, 
            HEIGHT = L4_HEIGHT, 
            WIDTH = L4_WIDTH, 
            INH_SIZE = 7
        )

        self.L23 = L23(
            net = net, 
            IN_CHANNEL = IN_CHANNEL, 
            OUT_CHANNEL = OUT_CHANNEL, 
            HEIGHT = L23_HEIGHT, 
            WIDTH = L23_WIDTH
        )

        self.input_layer = DataLoaderLayer(
            net=net,
            data_loader=dl,
            targets=target,
            widnow_size=Crop_Window_Height,
            saccades_on_each_image=SACCADE,
            rest_interval=50,
            train_iterations=TRAIN_ITERATIONS,
            phase_interval=REST_INTERVAL,
            test_iterations=TEST_ITERATIONS,
            train_images_number=TRAIN_IMAGES_NUMBER,
            test_images_number=TEST_IMAGES_NUMBER
        ).build_data_loader()

        self.fclayer = fullyConnected.FC(
            net = net, 
            N = 100, 
            K = 2
        )

        self.transformation = None
        
        ### connections
        self._add_connections()

    def _add_connections(self) :
        
        self.synapse_L4_L23 = Synapsis(
            net = self.net,
            src = self.L4.layer,
            dst = self.L23.layer,
            input_port="output",
            output_port="input",
            synapsis_behavior=prioritize_behaviors([
                SynapseInit(),
                AveragePool2D(current_coef = 50000),
            ]),
            synaptic_tag="Proximal"
        )


        self.synapse_INP_L56 = Synapsis(
            net = self.net,
            src = self.input_layer,
            dst = self.L56.layer,
            input_port = "data_out",
            output_port = "input",
            synapsis_behavior=prioritize_behaviors([
                SynapseInit(),]) | {
                275: LocationCoder()
            },
            synaptic_tag="Proximal"
        )

        self.synapse_INP_L4 = Synapsis(
            net = self.net,
            src = self.input_layer,
            dst = self.L4.layer,
            input_port="data_out",
            output_port="input",
            synapsis_behavior=prioritize_behaviors([
                SynapseInit(),
                WeightInitializer(weights = torch.normal(0.1, 2, (OUT_CHANNEL, IN_CHANNEL, KERNEL_HEIGHT, KERNEL_WIDTH)) ),
                Conv2dDendriticInput(current_coef = 20000 , stride = 1, padding = 0),
                Conv2dSTDP(a_plus=0.8, a_minus=0.001),
                WeightNormalization(norm = 4)
            ]),
            synaptic_tag="Proximal"
        )

        self.synapse_L23_FC = Synapsis(
            net = self.net,
            src = self.L23.layer,
            dst = self.fclayer.layer,
            input_port="output",
            output_port="input",
            synapsis_behavior=prioritize_behaviors([
                SynapseInit(),
                WeightInitializer(mode = "random"),
                SimpleDendriticInput(current_coef = 100),
                WeightNormalization(norm = 35),
                WeightClip(w_min = 0, w_max = 1.65)
            ]) | ({
                400 : AttentionBasedRSTDP(a_plus = 0.008 , a_minus = 0.001, tau_c = 6, attention_plus = 1, attention_minus = 0),
            }),
            synaptic_tag="Proximal"
        )

        self.synapse_L23_56 = Synapsis(
            net = self.net,
            src = self.L23.layer,
            dst = self.L56.layer,
            input_port="output",
            output_port="input",
            synaptic_tag="Apical, exi",
            synapsis_behavior=prioritize_behaviors(
                [
                    SynapseInit(), 
                    SimpleDendriticInput(),
                    WeightInitializer(mode="normal(0.05, 0.01)"),
                    SimpleSTDP(w_min=0, w_max=100, a_plus=1, a_minus=0.0008)
                ]
            )
        )

        self.synapse_L56_L23 = Synapsis(
            net = self.net,
            src = self.L56.layer,
            dst = self.L23.layer,
            input_port="output",
            output_port="input",
            synaptic_tag="Apical, exi",
            synapsis_behavior=prioritize_behaviors(
                [
                    SynapseInit(), 
                    SimpleDendriticInput(),
                    WeightInitializer(mode="normal(0.05, 0.01)"),
                    SimpleSTDP(w_min=0, w_max=10, a_plus=1, a_minus=0.0008)
                ]
            )
        )
        
    
    def inject_input(
        self,
        dataset: torch.Tensor,
        target: torch.Tensor,
        iterations: int,
        saccade_numbers: int = 1,
        shuffle: bool = True,
    ):
        if not self.transformation:
            self.transformation = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Grayscale(num_output_channels = 1), # not necessary
                    Conv2dFilter(DoGFilter(size = 5, sigma_1 = 4, sigma_2 = 1, zero_mean=True, one_sum=True).unsqueeze(0).unsqueeze(0)), # type: ignore
                    SqueezeTransform(dim = 0), # type: ignore
                ]
            )
        
        ### data prepration
        t = torch.arange(target.shape[0])
        
        if shuffle:
            np.random.shuffle(t)
            dataset = dataset[t]
            target = target[t]
        
        new_dataset = torch.empty(0, INPUT_HEIGHT, INPUT_WIDTH)
        
        for i in range(0, dataset.shape[0]):
            img = dataset[i]  # 4 in range [0, 5842) ; 9 in range [5842, 11791)
            img = Image.fromarray(img.numpy(), mode="L")
            img = self.transformation(img)
            new_dataset = torch.cat((new_dataset.data, img.data.view(1, *img.data.shape)), dim=0)

        if not self.net.behavior.get(5):
            self.net.add_behavior(5, behavior=SetTarget(target = target))
        
        dl = DataLoader(new_dataset,shuffle=False)
        
        ### inject data to cc
        input_layer = DataLoaderLayer(
            net=self.net,
            data_loader=dl,
            widnow_size=INPUT_HEIGHT,
            saccade_sizes = Crop_Window_Height,
            saccades_on_each_image=saccade_numbers,
            rest_interval=10,
            iterations=iterations
        ).build_data_loader()

        ### Missing Synapsis between a Layer and CoricalLayer
        
        Synapsis_Inp_L4 = Synapsis(
            net = self.net,
            src = input_layer,
            dst = self.L4,
            input_port="data_out",
            output_port="input",
            behavior=prioritize_behaviors([
                SynapseInit(),
                WeightInitializer(weights = torch.normal(0.1, 2, (OUT_CHANNEL, IN_CHANNEL, KERNEL_HEIGHT, KERNEL_WIDTH)) ),
                Conv2dDendriticInput(current_coef = 20000 , stride = 1, padding = 0),
                Conv2dSTDP(a_plus=0.3, a_minus=0.0008),
                WeightNormalization(norm = 4)
            ]),
            tag="Proximal"
        )

        Synapsis_Inp_L56 = Synapsis(
            net = self.net,
            src = input_layer,
            dst = self.L56,
            input_port = "data_out",
            output_port = "input",
            behavior=prioritize_behaviors([
                SynapseInit(),]) | {
                275: LocationCoder()
            },
            tag="Proximal"
        )
        return Synapsis_Inp_L4, Synapsis_Inp_L56
