import torch

from pymonntorch import *
from conex import *
from lateral_weight import LatheralWeight2Sparse

class L4() : 

    """
    Class of L4 Layer of cortical column, consisting of balanced network of exi and inh neuron group, \
        used for sensory feature extraction using Conv2D.
    """

    def __init__(self, net, HEIGHT, WIDTH, IN_CHANNEL, OUT_CHANNEL, INH_SIZE) : 

        """
        initialize the required parameters for L4 layer of cortical column and create the required synapses and neuron groups.

        note :  size of the layer should be set by calculating the size of the output of convolution layer according to 'padding', 'stride', input size and kernel_size.

        Args : 
            net (NetworkObject) : main network to connect L4 layer to.
            HEIGHT (int) : height of the neuron dimension after convolution.
            WIDTH (int) : width of the neuron dimension after convolution.
            IN_CHANNEL (int) : depth of the input given as input before convolution.
            OUT_CHANNEL (int) : depth of the neuron dimension after convolution.    
            INH_SIZE (int) : height and width of lateral inhibtion in each depth. the given should be an odd number.
        """

        self.net = net,
        self.net = self.net[0]
        self.config = {
            "HEIGHT" : HEIGHT,
            "WIDTH" : WIDTH,
            "IN_CHANNEL" : IN_CHANNEL,
            "OUT_CHANNEL" : OUT_CHANNEL,
            "J0" : 300,
            "p" : 0.8,
            "inh_shape" : (1, 1, OUT_CHANNEL + (1 - OUT_CHANNEL % 2), INH_SIZE, INH_SIZE)
        }

        self._add_ng()
        self._add_sg()
        self._add_layer()
    
    def _add_ng(self) :

        """
        adds exitatory and inhibitory neuron groups to L4, with size distribution of 80% to 20%.
        """

        J0 = self.config["J0"]
        p = self.config["p"]
        
        self.ng_e = NeuronGroup(
            size = NeuronDimension(depth = self.config["OUT_CHANNEL"] , height = self.config["HEIGHT"], width = self.config["WIDTH"]),
            net = self.net, 
            behavior = prioritize_behaviors([
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
                KWTA(k=10),
                ActivityBaseHomeostasis(window_size=10, activity_rate=200, updating_rate=0.0001),
                Fire(),
                SpikeTrace(tau_s = 15),
                NeuronAxon(),
            ])
        )

        self.ng_i = NeuronGroup(
            size = self.config["HEIGHT"] * self.config["WIDTH"] * self.config["OUT_CHANNEL"] // 4,
            net = self.net,
            tag = "inh", 
            behavior = prioritize_behaviors([
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
            ])
        )

    def _add_sg(self) :

        """
        adds synapses between exi and inh neuron groups to create a balanced network, it contains lateralinhibtion for exi neuron group.
        """
        
        J0 = self.config["J0"]
        p = self.config["p"]

        self.sg_ei = SynapseGroup(
            net = self.net,
            src = self.ng_e, 
            dst = self.ng_i,
            tag = "Proximal",
            behavior = prioritize_behaviors([
                SimpleDendriticInput(),
                SynapseInit(),
                WeightInitializer(mode = "ones", scale = J0/math.sqrt(1600 * p), density = 0.02, true_sparsity = False),
            ])
        )


        self.sg_ie = SynapseGroup(
            net = self.net,
            src = self.ng_i, 
            dst = self.ng_e, 
            tag = "Proximal", 
            behavior = prioritize_behaviors([
                SimpleDendriticInput(),
                SynapseInit(),
                WeightInitializer(mode = "ones", scale = J0/math.sqrt(1600 * p), density = 0.02, true_sparsity = False),
            ])
        )

        inh_lateral_weight = torch.ones(self.config["inh_shape"])
        inh_lateral_weight[0][0][self.config["inh_shape"][2] // 2][self.config["inh_shape"][3] // 2][self.config["inh_shape"][4] // 2] = 0

        self.sg_ee = SynapseGroup(
            net = self.net, 
            src = self.ng_e, 
            dst = self.ng_e, 
            tag = "Proximal", 
            behavior = prioritize_behaviors([
                SynapseInit(),
                WeightInitializer(weights=inh_lateral_weight),
                SimpleDendriticInput(current_coef=-20000),
            ]) | ({
                4 : LatheralWeight2Sparse(r_sparse=False)
            })
        )

        self.sg_ii = SynapseGroup(
            net = self.net, 
            src = self.ng_i,
            dst = self.ng_i, 
            tag = "Proximal", 
            behavior = prioritize_behaviors([
                SimpleDendriticInput(),
                SynapseInit(),
                WeightInitializer(mode = "ones", scale = J0/math.sqrt(1600 * p), density = 0.02, true_sparsity = False),
            ])
        )

    def _add_layer(self) : 
        
        """
        creates a CorticalLayer structure of L4 layer to be compatable as a component layer with other layers.
        """

        self.layer = CorticalLayer(
            net = self.net,
            excitatory_neurongroup = self.ng_e,
            inhibitory_neurongroup = self.ng_i,
            synapsegroups=[self.sg_ee, self.sg_ei, self.sg_ie, self.sg_ii],
            input_ports = {
                "input": (
                    None,
                    [Port(object = self.ng_e, label = None)],
                ),
            },
            output_ports = {
                "output": (
                    None,
                    [Port(object = self.ng_e, label = None)]
                )
            }
        )


class L23() :
    
    """
    Class of L23 Layer of cortical column, consisting of balanced network of exi and inh neuron group, 
        used for stable sensory feature representation using AveragePooling2D.
    """

    def __init__(self, net, HEIGHT, WIDTH, IN_CHANNEL, OUT_CHANNEL) :

        """
        initialize the required parameters for L23 layer of cortical column and create the required synapses and neuron groups.

        note :  size of the layer is calculated automatically by the size you give as input, no need to check the conv2d input width, height, kernel size and etc... 

        Args : 
            net (NetworkObject) : main network to connect L23 layer to.
            HEIGHT (int) : height of the neuron dimension.
            WIDTH (int) : width of the neuron dimension.
            IN_CHANNEL (int) : depth of the input given as input before convolution.
            OUT_CHANNEL (int) : depth of the neuron dimension after pooling.    
        """    

        self.net = net
        self.config = {
            "HEIGHT" : HEIGHT,
            "WIDTH" : WIDTH,
            "IN_CHANNEL" : IN_CHANNEL,
            "OUT_CHANNEL" : OUT_CHANNEL,
            "J0" : 300,
            "p" : 0.8,
        }

        self._add_ng()
        self._add_sg()
        self._add_layer()
    
    def _add_ng(self) : 

        """
        adds exitatory and inhibitory neuron groups to L23, with size distribution of 80% to 20%.
        """

        self.ng_e = NeuronGroup(
            size = NeuronDimension(depth = self.config["OUT_CHANNEL"] , height = self.config["HEIGHT"], width = self.config["WIDTH"]),
            net = self.net,
            behavior = prioritize_behaviors([
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
            ])
        )

        self.ng_i = NeuronGroup(
            size = self.config["HEIGHT"] * self.config["WIDTH"] * self.config["OUT_CHANNEL"] // 4,
            net = self.net,
            tag = "inh", 
            behavior = prioritize_behaviors([
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
            ])
        )

    def _add_sg(self) : 
        """
        adds synapses between exi and inh neuron groups to create a balanced network, it contains lateralinhibtion for exi neuron group.
        """
        J_0 = self.config["J0"]
        p = self.config["p"]

        self.sg_ei = SynapseGroup(
            net = self.net,
            src = self.ng_e, 
            dst = self.ng_i,
            tag = "Proximal",
            behavior = prioritize_behaviors([
                SimpleDendriticInput(),
                SynapseInit(),
                WeightInitializer(mode = "ones", scale = J_0/math.sqrt(2500 * p), density = 0.02, true_sparsity = False),
            ])
        )


        self.sg_ie = SynapseGroup(
            net = self.net,
            src = self.ng_i, 
            dst = self.ng_e, 
            tag = "Proximal", 
            behavior = prioritize_behaviors([
                SimpleDendriticInput(),
                SynapseInit(),
                WeightInitializer(mode = "ones", scale = J_0/math.sqrt(2500 * p), density = 0.02, true_sparsity = False),
            ])
        )

        self.sg_ii = SynapseGroup(
            net = self.net, 
            src = self.ng_i,
            dst = self.ng_i, 
            tag = "Proximal", 
            behavior = prioritize_behaviors([
                SimpleDendriticInput(),
                SynapseInit(),
                WeightInitializer(mode = "ones", scale = J_0/math.sqrt(2500 * p), density = 0.02, true_sparsity = False),
            ])
        )

    def _add_layer(self) : 

        """
        creates a CorticalLayer structure of L23 layer to be compatable as a component layer with other layers.
        """

        self.layer = CorticalLayer(
            net = self.net,
            excitatory_neurongroup = self.ng_e,
            inhibitory_neurongroup = self.ng_i,
            synapsegroups=[self.sg_ei, self.sg_ie, self.sg_ii],
            input_ports = {
                "input": (
                    None,
                    [Port(object = self.ng_e, label = None)],
                ),
            },
            output_ports = {
                "output": (
                    None,
                    [Port(object = self.ng_e, label = None)]
                )
            }
        )


class SensoryLayer() :

    """
    Works as the combination of L4 and L23 with the required synapses.

    note : still in development, DO NOT USE THIS.
    """

    def __init__(self, net, IN_CHANNEL, OUT_CHANNEL, L4_HEIGHT, L4_WIDTH, L23_HEIGHT, L23_WIDTH, INH_SIZE) : 
        
        self.net = net
        self.L4 = L4(net = net, IN_CHANNEL = IN_CHANNEL, OUT_CHANNEL = OUT_CHANNEL, HEIGHT = L4_HEIGHT, WIDTH = L4_WIDTH, INH_SIZE = INH_SIZE)
        self.L23 = L23(net = net, IN_CHANNEL = IN_CHANNEL, OUT_CHANNEL = OUT_CHANNEL, HEIGHT = L23_HEIGHT, WIDTH = L23_WIDTH)

        self._add_synapsis()

    def _add_synapsis(self) : 

        self.syn_L4_L23 = Synapsis(
            net = self.net,
            src = self.L4.layer,
            dst = self.L23.layer,
            input_port = "output",
            output_port = "input",
            synapsis_behavior=prioritize_behaviors([
                SynapseInit(),
                AveragePool2D(current_coef = 50000),
            ]),
            synaptic_tag="Proximal"
        )

        self.syn_L23_L4 = Synapsis(
            net = self.net,
            src = self.L4.layer,
            dst = self.L23.layer,
            input_port = "output",
            output_port = "input",
            synapsis_behavior=prioritize_behaviors([
                SynapseInit(),
                One2OneDendriticInput(current_coef = 10),
            ]),
            synaptic_tag="Apical"
        )

    def add_input_layer(self, input_layer) : 
        pass

    