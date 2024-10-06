from pymonntorch import *
from conex import *
import torch

from network.payoff import ConfidenceLevelPayOff

class FC() :
    """
    Fully Connected Layer works as the decision making layer (Classifier) using EI populations and competition.
    """

    def __init__(self, K, N, net = None, input_layer = None) : 
        """
        Creates the Network of Neural Poplations and their Synapses to be ready for the ouside input current.

        Args : 
            K (int): number of classes/excitatory NeuronGroups to be created.
            N (int): size of each neuron group (due to 80-20 exi-inh distribution, size of each exi population would be 0.8 * N)
            net (Network): An initial network to build the fully connected layer on it, if not determined, a new network will be created.
            input_layer (Container): Neural Layer to be connected to the FC layer. default is None, it can either be Layer, CortiacalColumn and etc...
        """

        self.N = N
        self.E = int(0.8 * N)
        self.I = int(0.2 * N)
        self.K = K


        self.net = net
        if(net == None) : 
            self.net = Network(behavior = prioritize_behaviors([
                TimeResolution(dt = 1),
                Dopamine(tau_dopamine = 20),
            ]) | ({
                100 : ConfidenceLevelPayOff()
            }))
        

        """
        add R-STDP configuration and Behaviors

        """
        net.add_behavior(prioritize_behaviors([Dopamine(tau_dopamine = 20)]) | ({100 : ConfidenceLevelPayOff()}))

        self.input_layer = input_layer

        """
        Network creation 
        """

        self.create_neuron_groups(N, K)
        self.create_synapses(K)
        self.create_layer()

        if(input_layer != None) : 
            self.create_input_connection(input_layer)


    def create_neuron_groups(self, N, K) : 

        """
        Creation of each EXI neuron group. & Creation of the single, shared INH neuron group.
        """

        self.E_NG_list = []
        for i in range(self.K) : 
            E_NG = NeuronGroup(net = self.net,
                size = self.E,
                behavior = prioritize_behaviors([
                    SimpleDendriteStructure(),
                    SimpleDendriteComputation(),
                    LIF(
                        R = 10,
                        tau = 5,
                        threshold = -10,
                        v_rest = -65,
                        v_reset = -67,
                        init_v =  -65,
                    ),
                    # InherentNoise(scale=random.randint(20, 60)),
                    Fire(),
                    NeuronAxon()
                ]) | ({ 
                    600 : Recorder(["I"]),
                    601 : EventRecorder(['spikes'])
                })
            )

            self.E_NG_list.append(E_NG)


        self.I_NG = NeuronGroup(net = self.net,
            size = self.I * self.K,
            tag = "inh",
            behavior = prioritize_behaviors([
                SimpleDendriteStructure(),
                SimpleDendriteComputation(),
                SpikeTrace(tau_s = 20),
                LIF(
                    R = 10,
                    tau = 5,
                    threshold = -10,
                    v_rest = -65,
                    v_reset = -67,
                    init_v =  -65,
                ),
                Fire(),
                NeuronAxon()
            ]) | ({ 
                600 : Recorder(["I"]),
                601 : EventRecorder(['spikes'])
            })
        )

    def create_synapses(self, K) : 

        """
        Creation of Synapses between each EXI to themselves and bidirectional connectivity to INH population.
        each excitatory neuron group has 3 main connections : 
            1. EE : exc to itself
            2. EI : exc to inh
            3. IE : inh to exc
        """

        for i in range(self.K) : 

            EE_SYN = SynapseGroup(
                net = self.net,
                src = self.E_NG_list[i], 
                dst = self.E_NG_list[i], 
                tag = "Proximal, EXI",
                behavior = prioritize_behaviors([
                    SynapseInit(),
                    WeightInitializer(mode = "random"),
                    SimpleDendriticInput(),
                ])
            )

            IE_SYN = SynapseGroup(
                net = self.net,
                src = self.I_NG, 
                dst = self.E_NG_list[i], 
                tag = "Proximal, inh",
                behavior = prioritize_behaviors([
                    SynapseInit(),
                    WeightInitializer(mode = 4),
                    SimpleDendriticInput(),
                ])
            )

            EI_SYN = SynapseGroup(
                net = self.net,
                src = self.E_NG_list[i], 
                dst = self.I_NG, 
                tag = "Proximal, EXI",
                behavior = prioritize_behaviors([
                    SynapseInit(),
                    WeightInitializer(mode = "random"),
                    SimpleDendriticInput(),
                ])
            )
        
    def create_layer(self) : 
        """
        Creating a Layer container for the whole decision-making network and their synapses between themselves.
        """

        self.layer = Layer(
            net = self.net,
            neurongroups = self.net.NeuronGroups,
            synapsegroups = self.net.SynapseGroups,
            input_ports = {
                "input" : (
                    None,
                    [Port(object=self.E_NG_list[i]) for i in range(self.K)]
                )
            }
        )


    def create_input_connection(self, input_layer) : 
        """
        Creating the synaptic connection between the input layer and the decision-making layer.
        """

        self.input_fc_connection = CorticalLayerConnection(
            net = self.net, 
            src = input_layer,
            dst = self.layer,
            connections = [
                (
                    "output",
                    "input",
                    prioritize_behaviors(
                        [
                            SynapseInit(),
                            WeightInitializer(),
                            SimpleDendriticInput(),
                            SimpleRSTDP(a_plus = 0.1 , a_minus = 0.002)
                        ]
                    ),
                    "Proximal",
                ),
            ],
        )
    
    def initialize(self, has_init = True) :
        self.net.initialize(has_init)

    def simulate_iterations(self, iterations) : 
        self.net.simulate_iterations(iterations)
