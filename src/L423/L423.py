import torch

from pymonntorch import *
from conex import *

### parameters
DoG_SIZE = 5
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

OUT_CHANNEL = 5
IN_CHANNEL = 1
KERNEL_WIDTH = 7
KERNEL_HEIGHT = 7

INPUT_WIDTH = IMAGE_WIDTH - DoG_SIZE + 1
INPUT_HEIGHT = IMAGE_HEIGHT - DoG_SIZE + 1

L4_WIDTH = INPUT_WIDTH - KERNEL_WIDTH + 1
L4_HEIGHT = INPUT_HEIGHT - KERNEL_HEIGHT + 1

L23_WIDTH = L4_WIDTH//2
L23_HEIGHT = L4_HEIGHT//2

J_0 = 300
p = 0.8


class SensoryLayer() :

    def __init__(
        self, 
        net,

    ) : 

        self.net = net
        self.layer = None
        self.create_L4()
        self.create_L23()
        self.connect_L4_L23()
        self.connect_input()

    def create_L4(self) : 
        
        self.ng4e = NeuronGroup(size = NeuronDimension(depth = OUT_CHANNEL , height = L4_HEIGHT, width = L4_WIDTH), net = self.net, behavior = prioritize_behaviors([
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
            KWTA(k=20),
            ActivityBaseHomeostasis(window_size=10, activity_rate=200, updating_rate=0.0001),
            Fire(),
            SpikeTrace(tau_s = 15),
            NeuronAxon(),
        ]))

        self.ng4i = NeuronGroup(size = L4_HEIGHT * L4_WIDTH * OUT_CHANNEL // 4, net = self.net, tag = "inh", behavior = prioritize_behaviors([
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
        ]))


        self.sg4e4i = SynapseGroup(net = self.net, src = self.ng4e, dst = self.ng4i, tag = "Proximal", behavior = prioritize_behaviors([
            SimpleDendriticInput(),
            SynapseInit(),
            WeightInitializer(mode = "ones", scale = J_0/math.sqrt(1600 * p), density = 0.02, true_sparsity = False),
        ]))


        self.sg4i4e = SynapseGroup(net = self.net, src = self.ng4i, dst = self.ng4e, tag = "Proximal", behavior = prioritize_behaviors([
            SimpleDendriticInput(),
            SynapseInit(),
            WeightInitializer(mode = "ones", scale = J_0/math.sqrt(1600 * p), density = 0.02, true_sparsity = False),
        ]))

        self.sg4e4e = SynapseGroup(net = self.net, src = self.ng4e, dst = self.ng4e, tag = "Proximal", behavior=prioritize_behaviors([
            SynapseInit(),
            WeightInitializer(weights=torch.Tensor([1, 1, 1, 1, 0, 1, 1, 1, 1]).view(1, 1, 9, 1, 1)),
            LateralDendriticInput(current_coef=100000, inhibitory = True),
        ]))

        self.sg4i4i = SynapseGroup(net = self.net, src = self.ng4i, dst = self.ng4i, tag = "Proximal", behavior = prioritize_behaviors([
            SimpleDendriticInput(),
            SynapseInit(),
            WeightInitializer(mode = "ones", scale = J_0/math.sqrt(1600 * p), density = 0.02, true_sparsity = False),
        ]))
        
        self.L4 = CorticalLayer(
            net=self.net,
            excitatory_neurongroup=self.ng4e,
            inhibitory_neurongroup=self.ng4i,
            synapsegroups=[self.sg4e4i, self.sg4i4e, self.sg4e4e, self.sg4i4i],
            input_ports={
                "input": (
                    None,
                    [Port(object = self.ng4e, label = None)],
                ),
                "output": (
                    None,
                    [Port(object = self.ng4e, label = None)]
                )
            },
        )

    def create_L23(self) : 

        ######################### L2&3 ########################

        self.ng23e = NeuronGroup(size = NeuronDimension(depth = OUT_CHANNEL , height = L23_HEIGHT, width = L23_WIDTH), net = self.net, behavior = prioritize_behaviors([
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
        ]))

        self.ng23i = NeuronGroup(size = L23_HEIGHT * L23_WIDTH * OUT_CHANNEL // 4, net = self.net, tag = "inh", behavior = prioritize_behaviors([
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
        ]))

        self.sg23e23i = SynapseGroup(net = self.net, src = self.ng23e, dst = self.ng23i, tag = "Proximal", behavior = prioritize_behaviors([
            SimpleDendriticInput(),
            SynapseInit(),
            WeightInitializer(mode = "ones", scale = J_0/math.sqrt(2500 * p), density = 0.02, true_sparsity = False),
        ]))

        self.sg23i23e = SynapseGroup(net = self.net, src = self.ng23i, dst = self.ng23e, tag = "Proximal", behavior = prioritize_behaviors([
            SimpleDendriticInput(),
            SynapseInit(),
            WeightInitializer(mode = "ones", scale = J_0/math.sqrt(2500 * p), density = 0.02, true_sparsity = False),
        ]))

        self.sg23i23i = SynapseGroup(net = self.net, src = self.ng23i, dst = self.ng23i, tag = "Proximal", behavior = prioritize_behaviors([
            SimpleDendriticInput(),
            SynapseInit(),
            WeightInitializer(mode = "ones", scale = J_0/math.sqrt(2500 * p), density = 0.02, true_sparsity = False),
        ]))


        ######################## Layers #######################

        self.L23 = CorticalLayer(
            net=self.net,
            excitatory_neurongroup=self.ng23e,
            inhibitory_neurongroup=self.ng23i,
            synapsegroups=[self.sg23e23i, self.sg23i23e, self.sg23i23i],
            input_ports={
                "input": (
                    None,
                    [Port(object = self.ng23e, label = None)],
                ),
                "output": (
                    None,
                    [Port(object = self.ng23e, label = None)]
                )
            },
        )


        ############### Inter Layer Connections ###############

    def connect_L4_L23(self) : 

        Synapsis_L4_L23 = Synapsis(
            net = self.net,
            src = self.L4,
            dst = self.L23,
            input_port="output",
            output_port="input",
            synapsis_behavior=prioritize_behaviors([
                SynapseInit(),
                AveragePool2D(current_coef = 50000),
            ]),
            synaptic_tag="Proximal"
        )


    def connect_input(self, input_layer) : 
        
        Synapsis_Inp_L4 = Synapsis(
            net = self.net,
            src = input_layer,
            dst = self.L4,
            input_port="data_out",
            output_port="input",
            synapsis_behavior=prioritize_behaviors([
                SynapseInit(),
                WeightInitializer(weights = torch.normal(0.1, 2, (OUT_CHANNEL, IN_CHANNEL, KERNEL_HEIGHT, KERNEL_WIDTH)) ),
                Conv2dDendriticInput(current_coef = 10000 , stride = 1, padding = 0),
                Conv2dSTDP(a_plus=0.3, a_minus=0.008),
                WeightNormalization(norm = 10)
            ]),
            synaptic_tag="Proximal"
        )