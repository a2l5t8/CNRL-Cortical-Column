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
from stimuli.current_base import RandomInputCurrent
from synapse.Weights import WeightInitializerAncher


pos_x, pos_y = generate_walk(length=100, R=10)

""" 
#######################################################
##################   CUSTOMIZATION   ##################
#######################################################
"""
layer_5_6_size = 24
higher_layer_size = 16


screen_shot_path = "/home/amilion/Pictures/Screenshots/GPCell"


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
            SimpleDendriteComputation(),
            Fire(),
            KWTA(k=25),
            NeuronAxon(),
        ]
    )
    | (
        {
            260: GPCell(
                R=10,
                tau=5,
                threshold=-60,
                v_rest=-65,
                v_reset=-67,
                init_v=-65,
                L=50,
                V=speed_vector_converter(pos_x, pos_y),
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
            250: RandomInputCurrent(prob_to_spike=0.1, T=50),
            # 600: Recorder(["I", "v"]),
            603: EventRecorder(["spikes"]),
        }
    ),
)

sg = SynapseGroup(
    net=net,
    src=higher_layer,
    dst=layer_5_6,
    tag="Proximal,exi",
    behavior=prioritize_behaviors(
        [
            SynapseInit(),
            SimpleDendriticInput(),
        ]
    )
    | (
        {
            280: WeightInitializerAncher(),
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

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = list(prop_cycle.by_key()["color"])


ngs = [higher_layer]
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
