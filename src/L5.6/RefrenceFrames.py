import torch

import conex as cnx

from pymonntorch import Recorder, EventRecorder, NeuronGroup, NeuronDimension, SynapseGroup

from synapse.GPCell_lateral_inhibition import GPCellLateralInhibition
from tools.rat_simulation import speed_vector_converter, generate_walk
from stimuli.current_base import ConstantCurrent
from neuron.GPCell import GPCell

class RefrenceFrame():
    """
    params: 
       (int) k: number of refrence frames
    """
    def __init__(
        self,
        k: int,
        refrence_frame_side: int,
        inhibitory_size: int,
        random_walk: bool = True,
        lateral_inhibition: bool = True,
        competize: bool = True,
        pos_x: list = None,
        pos_y: list = None,
        net: cnx.Neocortex = None
    ) -> None:
        self.net = net
        if not self.net:
            self.net = cnx.Neocortex(dt=1)
        self.pos_x, self.pos_y = pos_x, pos_y
        if random_walk:
            self.pos_x, self.pos_y = generate_walk(length=100, R=10)        
        self.k = k
        self.side = refrence_frame_side
        self.inh_side = inhibitory_size
        self.neuron_groups = []
<<<<<<< Updated upstream:src/L5.6/RefrenceFrames.py
=======
        self.input_neurons = []
        self.refrences = []
>>>>>>> Stashed changes:src/L56/RefrenceFrames.py
        self.synapse_groupes = []
        self.create_refrence_frames()
        # self.create_input_neurons()
        # self.input_neuron_to_refrences_syn()
        if competize:
            self.add_competition()
        if lateral_inhibition:
            self.add_lateral_inhibition()
        
        self.layer = self.build_layer()
<<<<<<< Updated upstream:src/L5.6/RefrenceFrames.py
        
=======
    
    def add_input_neuron(self, id: int):
        ng = cnx.NeuronGroup(
            net=self.net,
            size = 1,
            tag=f"InputRefrenceFrame, {id}",
            behavior=cnx.prioritize_behaviors(
                    [
                        cnx.SimpleDendriteStructure(),
                        cnx.SimpleDendriteComputation(apical_provocativeness=0.9),
                        cnx.LIF(R=8,
                            tau=5,
                            threshold=-40,
                            v_rest=-65,
                            v_reset=-67,),
                        cnx.Fire(),
                        cnx.KWTA(k=10),
                        cnx.NeuronAxon(),
                    ]
                ) | {
                    600 : Recorder(["v"]),
                    601 : EventRecorder(["spikes"])
                }
        )
        self.neuron_groups.append(ng)
        self.input_neurons.append(ng)
    
    def create_input_neurons(self):
        for ng_id in range(self.k):
            self.add_input_neuron(id=ng_id)
>>>>>>> Stashed changes:src/L56/RefrenceFrames.py
    
    def add_refrence_frame(self, id: int):
        ng = cnx.NeuronGroup(
                net=self.net,
                size=cnx.NeuronDimension(width=self.side, height=self.side),
                tag=f"RefrenceFrame,{id}",
                behavior=cnx.prioritize_behaviors(
                    [
                        cnx.SimpleDendriteStructure(),
                        cnx.SimpleDendriteComputation(apical_provocativeness=0.9),
                        cnx.Fire(),
<<<<<<< Updated upstream:src/L5.6/RefrenceFrames.py
                        # KWTA(k=10),
=======
                        cnx.KWTA(k=25),
                        cnx.SpikeTrace(tau_s = 5, offset = 0),
>>>>>>> Stashed changes:src/L56/RefrenceFrames.py
                        cnx.NeuronAxon(),
                    ]
                )
                | (
                    {
                        250: ConstantCurrent(scale=1.7),
                        260: GPCell(
                            R=8,
                            tau=5,
                            threshold=-30,
                            v_rest=-65,
                            v_reset=-67,
<<<<<<< Updated upstream:src/L5.6/RefrenceFrames.py
                            L=10,
                            I_amp = 20,
                            V=speed_vector_converter(self.pos_x, self.pos_y),
                            init_v=torch.tensor([-67]).expand(self.side * self.side).clone().to(dtype=torch.float32)
=======
                            L=5,
                            I_amp = 30,
                            init_v=torch.normal(-57, 10, size = (self.side**2, )).to(dtype=torch.float32)
>>>>>>> Stashed changes:src/L56/RefrenceFrames.py
                        ),
                        600: Recorder(["I", "v"]),
                        601: EventRecorder(["spikes"]),
                    }
                ) | ({600:Recorder(["spikes", "v", "_v"])}),
            )
        ng.gid = id
        self.neuron_groups.append(ng)

    def create_refrence_frames(self):
        for ng_id in range(self.k):
            self.add_refrence_frame(id=ng_id)
    
<<<<<<< Updated upstream:src/L5.6/RefrenceFrames.py
=======
    def input_neuron_to_refrences_syn(self):
        assert len(self.input_neurons) == len(self.refrences) == self.k
        for i in range(self.k):
            input_to_refrence = cnx.SynapseGroup(
                net=self.net,
                tag=f"input_to_refrence, Apical, {i}",
                src=self.input_neurons[i],
                dst=self.refrences[i],
                behavior = cnx.prioritize_behaviors(
                        [
                            cnx.SynapseInit(), cnx.SimpleDendriticInput(), cnx.WeightInitializer(mode="normal(2, 0)")
                        ]
                    )
            )
            self.synapse_groupes.append(input_to_refrence)
    
>>>>>>> Stashed changes:src/L56/RefrenceFrames.py
    def add_competion_syn(self, inhibitory: NeuronGroup):
        for neuron_group in self.neuron_groups:
            if inhibitory.tags == neuron_group.tags:
                syn_to_self = SynapseGroup(
                    net=self.net,
                    tag="inh_to_inh, Proximal",
                    src=inhibitory,
                    dst=neuron_group,
                    behavior=cnx.prioritize_behaviors(
                        [
<<<<<<< Updated upstream:src/L5.6/RefrenceFrames.py
                            cnx.SynapseInit(), cnx.SimpleDendriticInput(), cnx.WeightInitializer(mode="normal(1.5, 0.5)")
=======
                            cnx.SynapseInit(), cnx.SimpleDendriticInput(), cnx.WeightInitializer(mode="normal(0.7, 0.1)")
>>>>>>> Stashed changes:src/L56/RefrenceFrames.py
                        ]
                    )
                )
                self.synapse_groupes.append(syn_to_self)
                continue
            syn_from = SynapseGroup(
                net=self.net,
                tag=f"inh_to_refrence{neuron_group.tags[1]}, Proximal",
                src=inhibitory,
                dst=neuron_group,
                behavior=cnx.prioritize_behaviors(
                    [
<<<<<<< Updated upstream:src/L5.6/RefrenceFrames.py
                        cnx.SynapseInit(), cnx.SimpleDendriticInput(), cnx.WeightInitializer(mode="normal(1.5, 0.5)")
=======
                        cnx.SynapseInit(), cnx.SimpleDendriticInput(), cnx.WeightInitializer(mode="normal(0.8, 0.1)")
>>>>>>> Stashed changes:src/L56/RefrenceFrames.py
                    ]
                )
            )
            self.synapse_groupes.append(syn_from)
            syn_to = SynapseGroup(
                net=self.net,
                tag=f"refrence{neuron_group.tags[1]}_to_inh, Proximal",
                src=neuron_group,
                dst=inhibitory,
                behavior=cnx.prioritize_behaviors(
                    [
<<<<<<< Updated upstream:src/L5.6/RefrenceFrames.py
                        cnx.SynapseInit(), cnx.SimpleDendriticInput(), cnx.WeightInitializer(mode="normal(1.5, 0.5)")
=======
                        cnx.SynapseInit(), cnx.SimpleDendriticInput(), cnx.WeightInitializer(mode="normal(0.8, 0.1)")
>>>>>>> Stashed changes:src/L56/RefrenceFrames.py
                    ]
                )
            )
            self.synapse_groupes.append(syn_to)
            
            syn_ex_to_ex = SynapseGroup(
                net=self.net,
                tag=f"refrence{neuron_group.tags[1]}_to_itself, Proximal",
                src=neuron_group,
                dst=neuron_group,
                behavior=cnx.prioritize_behaviors(
                    [
                        cnx.SynapseInit(), cnx.SimpleDendriticInput(), cnx.WeightInitializer(mode="normal(0.01, 0.001)"), cnx.SimpleSTDP(a_plus=0.0007, a_minus=0.0001, w_min=0, w_max=0.2)
                    ]
                )
            )

            self.synapse_groupes.append(syn_ex_to_ex)
            
    def add_competition(self):
        inhibitory_neuron_group = NeuronGroup(
            net=self.net,
            size=self.inh_side,
            tag="inh",
            behavior=cnx.prioritize_behaviors(
                [
                    cnx.SimpleDendriteStructure(),
                    cnx.SimpleDendriteComputation(),
                    cnx.LIF(
                        R=10,
                        tau=8,
                        v_rest=-63,
                        v_reset=-65,
                        threshold=-50
                    ),
                    cnx.Fire(),
                    cnx.NeuronAxon(),
                ]
            )
        )
        self.neuron_groups.append(inhibitory_neuron_group)
        self.add_competion_syn(inhibitory=inhibitory_neuron_group)
    
    
    def add_lateral_inhibition(self):
        for neuron_group in self.neuron_groups:
            if "inh" in neuron_group.tags:
                continue
            syn_group = SynapseGroup(
                net=self.net,
                tag=f"Lateral,Proximal,{neuron_group.tags[1]}", 
                src=neuron_group,
                dst=neuron_group,
                behavior=cnx.prioritize_behaviors(
                    [cnx.SynapseInit(), cnx.LateralDendriticInput()]
                )
                | (
                    {
<<<<<<< Updated upstream:src/L5.6/RefrenceFrames.py
                        180: GPCellLateralInhibition(kernel_side=31, max_inhibition=3, r=16, n=5, inhibitory=1),
=======
                        3: GPCellLateralInhibition(kernel_side=31, max_inhibition=3, r=16, n=5, inhibitory=1),
>>>>>>> Stashed changes:src/L56/RefrenceFrames.py
                    }
                ),
            )
            self.synapse_groupes.append(syn_group)
    
    def build_layer(self):
<<<<<<< Updated upstream:src/L5.6/RefrenceFrames.py
        return cnx.Layer(
            net=self.net,
            neurongroups=self.neuron_groups,
            synapsegroups=self.synapse_groupes,
            tag="layer_5_6"
=======
        input_ports = {
            "input" : (None, [cnx.Port(object = reference, label = None) for reference in self.refrences]),
        }
        for input_neuron in self.input_neurons:
            input_ports.update({
                f"general_input_to_input{input_neuron.tags[1]}" : (None, [cnx.Port(object=input_neuron, label=None)])         
            })
        for refrences in self.refrences:
            input_ports.update({
                f"general_input_to_reference{refrences.tags[1]}" : (None, [cnx.Port(object=refrences, label=None)])         
            })
        layer = cnx.Layer(
            net=self.net,
            neurongroups=self.neuron_groups,
            synapsegroups=self.synapse_groupes,
            tag="layer_5_6",
            input_ports= input_ports,
            output_ports= {
                "output" : 
                    (None, [cnx.Port(object = reference, label = None) for reference in self.refrences]),
            },
            # behavior={
            #     255 : PunishModulatorCurrent(group="RefrenceFrame", base_line=20, punish=-10, decay_tau=5),
            # }
>>>>>>> Stashed changes:src/L56/RefrenceFrames.py
        )








