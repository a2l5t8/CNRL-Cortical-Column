""" 
#######################################################
#####################   IMPORTS   #####################
#######################################################
"""

import numpy as np
from matplotlib import pyplot as plt
import random
import math
import pandas as pd
import tqdm
import seaborn as sns

from pymonntorch import *
from conex import *


from tools.rat_simulation import generate_walk, speed_vector_converter
from tools.visualization import iter_spike_multi_real

from neuron.GPCell import GPCell
from stimuli.current_base import RandomInputCurrent, ConstantCurrent

from synapse.GPCell_lateral_inhibition import GPCellLateralInhibition


pos_x, pos_y = generate_walk(length=100, R=10)

""" 
#######################################################
##################   CUSTOMIZATION   ##################
#######################################################
"""
layer_5_6_size = 24
higher_layer_size = 16


screen_shot_path = "C:\\Users\\amilion\\Desktop\\develop\\python\\NS\\records\\L5.6"


""" 
#######################################################
#################   Network Creation   ################
#######################################################
"""

net = Network(behavior=prioritize_behaviors([TimeResolution(dt=1)]))

layer_5_6 = NeuronGroup(
    net=net,
    size=NeuronDimension(width=layer_5_6_size, height=layer_5_6_size),
    behavior=prioritize_behaviors(
        [
            SimpleDendriteStructure(),
            SimpleDendriteComputation(apical_provocativeness=0.9),
            Fire(),
            # KWTA(k=10),
            NeuronAxon(),
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
                V=speed_vector_converter(pos_x, pos_y),
                init_v=torch.tensor([-67]).expand(layer_5_6_size * layer_5_6_size).clone().to(dtype=torch.float32)
            ),
            600: Recorder(["I", "v"]),
            601: EventRecorder(["spikes"]),
        }
    ),
)

higher_layer = NeuronGroup(
    net=net,
    size=NeuronDimension(depth=1, height=higher_layer_size, width=higher_layer_size),
    behavior=prioritize_behaviors(
        [
            SimpleDendriteStructure(),
            SimpleDendriteComputation(),
            NeuronAxon(),
        ]
    )
    | (
        {
            250: RandomInputCurrent(prob_to_spike=0.1, T=10),
            # 600: Recorder(["I", "v"]),
            603: EventRecorder(["spikes"]),
        }
    ),
)

sg = SynapseGroup(
    net=net,
    src=higher_layer,
    dst=layer_5_6,
    tag="Apical, exi",
    behavior=prioritize_behaviors(
        [SynapseInit(), SimpleDendriticInput() ,WeightInitializer(mode="normal(0.2, 3)")]
    )
    # | (
    #     {
    #         180: GPCellLateralInhibition(max_inhibition=3, r=3, n=9),
    #     }
    # ),
)

GP_lateral = SynapseGroup(
    net=net,
    src=layer_5_6,
    dst=layer_5_6,
    tag="Proximal",
    behavior=prioritize_behaviors(
        [SynapseInit(), SimpleDendriticInput()]
    )
    | (
        {
            180: GPCellLateralInhibition(kernel_side=31, max_inhibition=3, r=16, n=5, inhibitory=1),
        }
    ),
)


net.initialize()
net.simulate_iterations(101)

""" 
#######################################################
##################   Visualization   ##################
#######################################################
"""

# print(layer_5_6["I"])

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = list(prop_cycle.by_key()["color"])


ngs = [layer_5_6]
for i in range(0, 100):
    cnt = 0
    for ng in ngs:
        iter_spike_multi_real(
            pos_x,
            pos_y,
            ng,
            itr=i,
            step=1,
            color=colors[cnt],
            save=True,
            lib=screen_shot_path,
            label="GPCell" + str(cnt + 1),
            offset_x=0,
            offset_y=0,
            base_offset_x=0,
            base_offset_y=0,
        )
        cnt += 1
        # break

    plt.clf()
