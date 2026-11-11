"""L5/6 reference frames and their fixed sensorimotor input circuit."""

import torch
from conex import (
    LIF,
    Fire,
    NeuronAxon,
    One2OneDendriticInput,
    PreSpikeCatcher,
    SimpleDendriteComputation,
    SimpleDendriteStructure,
    SynapseInit,
    WeightInitializer,
    prioritize_behaviors,
)
from pymonntorch import NeuronGroup, SynapseGroup

from .behavior.neurons import (
    IF,
    BindingClamp,
    ExternalSpikes,
    MarginWTA,
    SensorimotorCoincidence,
)
from .behavior.specs import SpikeTrace
from .behavior.synapse import DenseDendriticInput, ImmediateIdentityInput


class L56:
    def __init__(self, net, classes, grid=3):
        side = grid + 2
        self.frame_size = side**2 + 1
        self.object_locations = [
            (row + 1) * side + col + 1 for row in range(grid) for col in range(grid)
        ]
        self.net = net
        self.classes = classes
        device = net.device
        previous = set([*net.NeuronGroups, *net.SynapseGroups])

        self.motor = NeuronGroup(
            net=net,
            size=grid**2,
            tag="Motor",
            behavior={340: ExternalSpikes(), 380: NeuronAxon()},
        )
        self.cue = NeuronGroup(
            net=net,
            size=classes,
            tag="AnchorCue",
            behavior={340: ExternalSpikes(), 380: NeuronAxon()},
        )
        self.hypotheses = NeuronGroup(
            net=net,
            size=classes,
            tag="FrameHypotheses",
            behavior=prioritize_behaviors(
                [
                    SimpleDendriteStructure(Apical_max_delay=1, apical_min_delay=0),
                    SimpleDendriteComputation(),
                    IF(),
                    MarginWTA(margin=float("inf")),
                    NeuronAxon(),
                ]
            ),
        )
        behaviors = prioritize_behaviors(
            [
                SimpleDendriteStructure(Apical_max_delay=1, apical_min_delay=0),
                SimpleDendriteComputation(apical_provocativeness=0.9),
                LIF(R=1, tau=1, threshold=0.5, v_reset=0, v_rest=0),
                Fire(),
                NeuronAxon(),
            ]
        )
        behaviors.update({350: BindingClamp(), 360: SpikeTrace()})
        self.neurons = NeuronGroup(
            net=net,
            size=classes * self.frame_size,
            tag="L56",
            behavior=behaviors,
        )
        self.path_relays = NeuronGroup(
            net=net,
            size=classes * self.frame_size,
            tag="SensorimotorRelays",
            behavior={190: SensorimotorCoincidence(), 380: NeuronAxon()},
        )

        self._connect(
            self.cue,
            self.hypotheses,
            One2OneDendriticInput(),
            torch.tensor(1e6, device=device),
        )
        hyp_weights = torch.zeros(classes, classes * self.frame_size, device=device)
        motor_weights = torch.zeros(grid**2, classes * self.frame_size, device=device)
        for frame in range(classes):
            for slot, location in enumerate(self.object_locations):
                target = frame * self.frame_size + location
                hyp_weights[frame, target] = 1
                motor_weights[slot, target] = 1

        self._connect(
            self.hypotheses, self.path_relays, DenseDendriticInput(), hyp_weights
        )
        self._connect(
            self.motor, self.path_relays, DenseDendriticInput(), motor_weights
        )
        SynapseGroup(
            net=net,
            src=self.path_relays,
            dst=self.neurons,
            tag="Proximal",
            behavior={2: SynapseInit(), 200: ImmediateIdentityInput()},
        )
        self.objects = [
            obj
            for obj in [*net.NeuronGroups, *net.SynapseGroups]
            if obj not in previous
        ]

    def _connect(self, source, destination, drive, weights):
        return SynapseGroup(
            net=self.net,
            src=source,
            dst=destination,
            tag="Proximal",
            behavior=prioritize_behaviors(
                [
                    SynapseInit(),
                    PreSpikeCatcher(),
                    drive,
                    WeightInitializer(weights=weights),
                ]
            ),
        )
