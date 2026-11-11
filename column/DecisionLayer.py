"""Decision relays, explicit inhibition and spike-count readout."""

import torch
from conex import SynapseInit
from pymonntorch import NeuronGroup, SynapseGroup

from .behavior.neurons import BurstIF, Coincidence, InputSpikes
from .behavior.synapse import BroadcastSynapse, CenteredRSTDP, DenseCurrent
from .simulation import activate


class DecisionLayer:
    """Framework graph; wrappers only supply stimuli and schedule simulation."""

    def __init__(
        self,
        net,
        features,
        classes,
        k,
        *,
        lr=0.003,
        noise=0.003,
        gain=None,
        a_minus=0.1,
        seed=42,
        grid=3,
        position=True,
    ):
        device = net.device
        self.position = position
        self.F, self.C, self.k, self.Q = (
            features,
            classes,
            k,
            grid**2 if position else 1,
        )
        self.device, self.noise, self.seed = device, noise, seed
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.net = net
        prior = set([*net.NeuronGroups, *net.SynapseGroups])
        net.phase = "idle"
        net.lr, net.gain, net.a_minus = lr, 0.9 / k if gain is None else gain, a_minus
        net.learning = False
        self.features = NeuronGroup(
            net=net, size=features, tag="L23Input", behavior={100: InputSpikes()}
        )
        if position:
            self.locations = NeuronGroup(
                net=net, size=self.Q, tag="L56Input", behavior={100: InputSpikes()}
            )
        self.relay = NeuronGroup(
            net=net,
            size=self.Q * features,
            tag="Coincidence",
            behavior={160: Coincidence(threshold=1.5 if position else 0.5)},
        )
        indices = torch.arange(self.Q * features, device=device)
        inputs = [(self.features, indices % features)]
        if position:
            inputs.append((self.locations, indices // features))
        for source, projection in inputs:
            SynapseGroup(
                net=net,
                src=source,
                dst=self.relay,
                tag="Proximal",
                behavior={2: SynapseInit(), 140: BroadcastSynapse(indices=projection)},
            )
        self.interneurons = NeuronGroup(
            net=net,
            size=self.Q * features,
            tag="InhibitoryRelay",
            behavior={175: Coincidence(threshold=0.5)},
        )
        SynapseGroup(
            net=net,
            src=self.relay,
            dst=self.interneurons,
            tag="Proximal",
            behavior={2: SynapseInit(), 170: BroadcastSynapse(indices=indices)},
        )
        self.neurons = NeuronGroup(
            net=net, size=classes, tag="Decision", behavior={260: BurstIF()}
        )
        self.synapse = SynapseGroup(
            net=net,
            src=self.relay,
            dst=self.neurons,
            tag="Proximal",
            behavior={2: SynapseInit(), 180: DenseCurrent(), 500: CenteredRSTDP()},
        )
        self.synapse.weights = (
            0.5
            + torch.rand(
                self.Q * features, classes, device=device, generator=self.generator
            )
            * 0.001
        )
        self.inhibition = SynapseGroup(
            net=net,
            src=self.interneurons,
            dst=self.neurons,
            tag="Proximal",
            behavior={2: SynapseInit(), 180: DenseCurrent(sign=-1.0)},
        )
        self.inhibition.weights = torch.full_like(self.synapse.weights, 0.5)
        self.objects = [
            obj for obj in [*net.NeuronGroups, *net.SynapseGroups] if obj not in prior
        ]

    @property
    def w(self):
        return self.synapse.weights

    @w.setter
    def w(self, value):
        self.synapse.weights = value

    @property
    def counts(self):
        return self.neurons.counts

    def reset(self, batch, learning=True):
        activate(self.net, self.objects)
        self.net.learning = learning
        self.neurons.v = self.w.new_zeros(batch, self.C)
        self.neurons.counts = torch.zeros_like(self.neurons.v)
        self.neurons.ticks = 0
        self.neurons.bias_noise = (
            torch.randn(batch, self.C, device=self.device, generator=self.generator)
            * self.noise
            if learning
            else torch.zeros_like(self.neurons.v)
        )
        self.synapse.pre_trace = self.w.new_zeros(batch, self.Q * self.F)
        self.synapse.post_trace = self.w.new_zeros(batch, self.C)
        self.synapse.expected_trace = torch.zeros_like(self.synapse.post_trace)
        self.synapse.eligibility = (
            self.w.new_zeros(batch, self.Q * self.F, self.C) if learning else None
        )
        self.awaiting_feedback = learning

    def present(self, spikes, slot, steps=64):
        self.features.external = spikes.bool()
        if self.position:
            self.locations.external = torch.zeros(
                len(spikes), self.Q, dtype=torch.bool, device=self.device
            )
            self.locations.external[:, slot] = True
        self.net.window = steps
        self.net.slot = slot
        self.net.phase = "sense"
        self.net.simulate_iterations(1, measure_block_time=False)
        self.net.phase = "idle"

    @torch.no_grad()
    def trial(self, drives, schedule, learning=True):
        self.reset(len(drives), learning)
        curves = []
        for slot, steps in enumerate(schedule):
            values, indices = drives[:, slot].topk(self.k, dim=1, sorted=False)
            spikes = torch.zeros_like(drives[:, slot]).scatter_(
                1, indices, (values >= 12.5).float()
            )
            self.present(spikes, slot, steps)
            curves.append(self.counts.argmax(1))
        return torch.stack(curves, 1)

    def release(self):
        if self.net.learning:
            top = self.counts == self.counts.max(1, keepdim=True).values
            draw = torch.rand(
                self.counts.shape, device=self.device, generator=self.generator
            )
            return draw.masked_fill(~top, -1).argmax(1)
        return self.counts.argmax(1)

    @torch.no_grad()
    def feedback(self, reward):
        if not self.awaiting_feedback:
            raise RuntimeError("Feedback requires an unrewarded training trial")
        self.net.reward = reward.to(self.device)
        self.net.phase = "feedback"
        self.net.simulate_iterations(1, measure_block_time=False)
        self.net.phase = "idle"
        self.awaiting_feedback = False
        return self.net.dopamine

    def state_dict(self):
        return dict(
            F=self.F,
            C=self.C,
            k=self.k,
            grid=int(self.Q**0.5),
            lr=self.net.lr,
            noise=self.noise,
            gain=self.net.gain,
            a_minus=self.net.a_minus,
            seed=self.seed,
            w=self.w.cpu(),
            reward_baseline=self.net.baseline,
            generator=self.generator.get_state(),
        )
