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
        self.synapse_groupes = []
        self.create_refrence_frames()
        if competize:
            self.add_competition()
        if lateral_inhibition:
            self.add_lateral_inhibition()
        
        self.layer = self.build_layer()
        
    
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
                        # KWTA(k=10),
                        cnx.NeuronAxon(),
                    ]
                )
                | (
                    {
                        250: ConstantCurrent(scale=4),
                        260: GPCell(
                            R=8,
                            tau=5,
                            threshold=-30,
                            v_rest=-65,
                            v_reset=-67,
                            L=10,
                            I_amp = 20,
                            V=speed_vector_converter(self.pos_x, self.pos_y),
                            init_v=torch.tensor([-67]).expand(self.side * self.side).clone().to(dtype=torch.float32)
                        ),
                        600: Recorder(["I", "v"]),
                        601: EventRecorder(["spikes"]),
                    }
                ),
            )
        self.neuron_groups.append(ng)

    def create_refrence_frames(self):
        for ng_id in range(self.k):
            self.add_refrence_frame(id=ng_id)
    
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
                            cnx.SynapseInit(), cnx.SimpleDendriticInput(), cnx.WeightInitializer(mode="normal(1.5, 0.5)")
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
                        cnx.SynapseInit(), cnx.SimpleDendriticInput(), cnx.WeightInitializer(mode="normal(1.5, 0.5)")
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
                        cnx.SynapseInit(), cnx.SimpleDendriticInput(), cnx.WeightInitializer(mode="normal(1.5, 0.5)")
                    ]
                )
            )
            self.synapse_groupes.append(syn_to)
            

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
                    [cnx.SynapseInit(), cnx.SimpleDendriticInput()]
                )
                | (
                    {
                        180: GPCellLateralInhibition(kernel_side=31, max_inhibition=3, r=16, n=5, inhibitory=1),
                    }
                ),
            )
            self.synapse_groupes.append(syn_group)
    
    def build_layer(self):
        return cnx.Layer(
            net=self.net,
            neurongroups=self.neuron_groups,
            synapsegroups=self.synapse_groupes,
            tag="layer_5_6"
        )








