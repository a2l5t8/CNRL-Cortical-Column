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

from neuron.GPCell import SampleAncher, GPCell
from synapse.Weights import WeightInitializerAncher


pos_x, pos_y = generate_walk(length=100, R=10)

""" 
#######################################################
##################   CUSTOMIZATION   ##################
#######################################################
"""
layer_5_6_size = 24
higher_layer_size = 16


screen_shot_path = r"../../res/apical_syn"


""" 
#######################################################
#################   Network Creation   ################
#######################################################
"""

net = Network(behavior=prioritize_behaviors([TimeResolution(dt=1)]))

ng = NeuronGroup(
    net=net,
    size=NeuronDimension(width=N, height=N),
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

inp = NeuronGroup(
    net=net,
    size=1,
    behavior=prioritize_behaviors(
        [SimpleDendriteStructure(), SimpleDendriteComputation(), NeuronAxon()]
    )
    | (
        {
            260: SampleAncher(),
            # 600 : Recorder(["I", "v"]),
            # 603 : EventRecorder(['spikes'])
        }
    ),
)

sg = SynapseGroup(
    net=net,
    src=inp,
    dst=ng,
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


ngs = [ng]
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
            lib="/home/amilion/Pictures/Screenshots/GPCell",
            label="GPCell" + str(cnt + 1),
            offset_x=0,
            offset_y=0,
            base_offset_x=0,
            base_offset_y=0,
        )
        cnt += 1
        # break

    plt.clf()
