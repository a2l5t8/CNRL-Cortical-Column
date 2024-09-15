from pymonntorch import *
from conex import *
import torch

class FC() :
    """
    Fully Connected Layer works as the decision making layer (Classifier) using EI populations and competition.
    """

    def __init__(self, K, N) : 
        """
        Creates the Network of Neural Poplations and their Synapses to be ready for the ouside input current.

        Args : 
            (int) K: number of classes/excitatory NeuronGroups to be created.
            (int) N: size of each neuron group (due to 80-20 exi-inh distribution, 
                                                size of each exi population would be 0.8 * N)
            (Layer) INP : Neural Layer to be connected to the FC layer.
        """

        self.N = N
        self.E = int(0.8 * N)
        self.I = int(0.2 * N)
        self.K = K




        self.net = Network(behavior = prioritize_behaviors([
            TimeResolution(dt = 1)
        ]))


        """
        Creation of each EXI neuron group.
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
                    InherentNoise(scale=random.randint(20, 60)),
                    Fire(),
                    NeuronAxon()
                ]) | ({ 
                    600 : Recorder(["I"]),
                    601 : EventRecorder(['spikes'])
                })
            )

            self.E_NG_list.append(E_NG)

        """
        Creation of the single, shared INH neuron group.
        """

        self.I_NG = NeuronGroup(net = self.net,
            size = self.I * self.K,
            tag = "inh",
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
                Fire(),
                NeuronAxon()
            ]) | ({ 
                600 : Recorder(["I"]),
                601 : EventRecorder(['spikes'])
            })
        )

        """
        Creation of Synapses between each EXI to themselves and bidirectional connectivity to INH population.
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
    
    def initialize(self, has_init = True) :
        self.net.initialize(has_init)

    def simulate_iterations(self, iterations) : 
        self.net.simulate_iterations(iterations)
