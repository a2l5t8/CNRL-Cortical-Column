# #### -- PROJECT PATH -- ####
# import sys
# sys.path.append(r'C:\Users\amilion\Documents\GitHub\CNRL-Cortical-Column')

#### -- IMPORTS -- ####

from matplotlib import pyplot as plt


from pymonntorch import *
from conex import *

from src.InputLayer.stimuli.OnlineDataLoader import OnlineDataLoader
from src.InputLayer.synapse.LocationCoder import LocationCoder

from src.L56.neuron.GPCell import GPCell
from src.L56.stimuli.current_base import ConstantCurrent, RandomInputCurrent
from src.L56.synapse.GPCell_lateral_inhibition import GPCellLateralInhibition

from src.L56.tools.rat_simulation import speed_vector_converter, generate_walk
from src.L56.tools.visualization import iter_spike_multi_real

#### -- PARAMETERS -- ####
image_size = 28
image_numbers = 5
saccades_on_each_image = 5
iterations = 1000
layer_5_6_size = 24
higher_layer_size = 16
# pos_x, pos_y = generate_walk(length=100, R=10)
screen_shot_path = "C:\\Users\\amilion\\Desktop\\develop\\python\\NS\\records\\L5.6"


#### -- NETWORK INTIALIZING -- ####

net = Network(behavior=prioritize_behaviors([TimeResolution(dt=1)]))

#### -- NEURON GROUP INTIALIZING -- ####
layer_5_6 = NeuronGroup(
    net=net,
    size=NeuronDimension(width=layer_5_6_size, height=layer_5_6_size),
    behavior=prioritize_behaviors(
        [
            SimpleDendriteStructure(),
            SimpleDendriteComputation(apical_provocativeness=0.9),
            Fire(),
            KWTA(k=10),
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
                L=5,
                I_amp = 20,
                # V=speed_vector_converter(pos_x, pos_y),
                init_v=torch.tensor([-67]).expand(layer_5_6_size * layer_5_6_size).clone().to(dtype=torch.float32)
            ),
            600: Recorder(["I", "v"]),
            601: EventRecorder(["spikes"]),
        }
    ),
)

loader_neuron_group = NeuronGroup(
    net=net,
    size=NeuronDimension(depth=1, height=image_size, width=image_size),
    behavior=prioritize_behaviors(
        [
            SimpleDendriteStructure(),
            SimpleDendriteComputation(),
            LIF(
                R=10,
                tau=5, 
                v_reset=-67,
                v_rest=-67,
                threshold=-60,
            ),
            Fire(),
            NeuronAxon(),
        ]) | {
            270: OnlineDataLoader(
                data_set=torch.rand(image_numbers, image_size, image_size), 
                batch_number=saccades_on_each_image,
                iterations=iterations
            ),
            600: Recorder(["focus_loc"]),
            601: EventRecorder(["spikes"]),
        }
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

#### -- SYNAPSE GROUP INTIALIZING -- ####


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

Loader_to_GP = SynapseGroup(
    net=net,
    src=loader_neuron_group,
    dst=layer_5_6,
    behavior={
        275: LocationCoder(),
    }
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
            180: GPCellLateralInhibition(kernel_side=29, max_inhibition=2, r=17, n=1.1, inhibitory=1),
        }
    ),
)




#### -- RUNNING THE MODEL -- ####

net.initialize()
net.simulate_iterations(iterations)

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
            loader_neuron_group["focus_loc"][0],
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

