import torch

import conex as cnx

# from pymonntorch import Recorder, EventRecorder, NeuronGroup, NeuronDimension, SynapseGroup
from pymonntorch import *

from L56.synapse.GPCell_lateral_inhibition import GPCellLateralInhibition
from L56.tools.rat_simulation import speed_vector_converter, generate_walk
from L56.stimuli.current_base import ConstantCurrent
from L56.neuron.GPCell import GPCell
from L56.synapse.vDistributor import vDistributor

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
        lateral_inhibition: bool = True,
        competize: bool = True,
        net: cnx.Neocortex = None
    ) -> None:
        self.net = net
        if not self.net:
            self.net = cnx.Neocortex(dt=1)        
        self.k = k
        self.side = refrence_frame_side
        self.inh_side = inhibitory_size
        self.neuron_groups = []
        self.refrences = []
        self.synapse_groupes = []
        self.create_refrence_frames()
        if competize:
            self.add_competition()
        if lateral_inhibition:
            self.add_lateral_inhibition()
        # self.add_input_neuron()
        self.layer = self.build_layer()
        # import pdb;pdb.set_trace()
    
    def add_input_neuron(self):
        ng = cnx.NeuronGroup(
            net=self.net,
            size = 1,
            tag=f"InputRefrenceFrame",
            behavior=cnx.prioritize_behaviors(
                    [
                        cnx.SimpleDendriteStructure(),
                        cnx.SimpleDendriteComputation(apical_provocativeness=0.9),
                        cnx.Fire(),
                        cnx.KWTA(k=10),
                        cnx.NeuronAxon(),
                    ]
                )
        )
        self.neuron_groups.append(ng)
    
    def add_refrence_frame(self, id: int):
        ng = cnx.NeuronGroup(
                net=self.net,
                size=NeuronDimension(width=self.side, height=self.side),
                tag=f"RefrenceFrame,{id}",
                behavior=cnx.prioritize_behaviors(
                    [
                        cnx.SimpleDendriteStructure(),
                        cnx.SimpleDendriteComputation(apical_provocativeness=0.9),
                        cnx.Fire(),
                        # cnx.KWTA(k=10),
                        cnx.NeuronAxon(),
                    ]
                )
                | (
                    {
                        250: ConstantCurrent(scale=10),
                        260: GPCell(
                            R=8,
                            tau=5,
                            threshold=-40,
                            v_rest=-65,
                            v_reset=-67,
                            L=10,
                            I_amp = 20,
                            init_v=torch.tensor([-67]).expand(self.side * self.side).clone().to(dtype=torch.float32)
                        ),
                        600: Recorder(["I", "v"]),
                        601: EventRecorder(["spikes"]),
                    }
                ),
            )
        self.neuron_groups.append(ng)
        self.refrences.append(ng)

    def create_refrence_frames(self):
        for ng_id in range(self.k):
            self.add_refrence_frame(id=ng_id)
    
    def input_neuron_to_refrences_syn(self):
        for refrence_frame in self.neuron_groups:
            if "RefrenceFrame" in refrence_frame.tags:
                input_to_refrence = cnx.SynapseGroup(
                    net = self.net,
                    tag = "input_to_refrence, Proximal",
                    src = self.neuron_groups[-1],
                    dst = refrence_frame,
                    behavior = 
                    {
                        275: vDistributor()
                    }
                )
                self.synapse_groupes.append(input_to_refrence)
    
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
                            cnx.SynapseInit(), cnx.SimpleDendriticInput(), cnx.WeightInitializer(mode="normal(1, 0.5)")
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
                        cnx.SynapseInit(), cnx.SimpleDendriticInput(), cnx.WeightInitializer(mode="normal(1, 0.5)")
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
                        cnx.SynapseInit(), cnx.SimpleDendriticInput(), cnx.WeightInitializer(mode="normal(1, 0.5)")
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
            tag="layer_5_6",
            input_ports= {"input": (None, list(map(lambda x: cnx.Port(object = x, label = None), self.refrences)))}
        )
        
    def cosine_similarity_test(self, first_feature: torch.Tensor, second_feature: torch.Tensor):
        import pdb;pdb.set_trace()
    







