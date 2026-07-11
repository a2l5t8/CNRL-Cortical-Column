"""Custom behaviors attached to cortical-column synapse groups."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from pymonntorch import Behavior, SynapseGroup

__all__ = [
    "MotorToL56Current",
    "SpikingConvolutionInput",
    "BatchedConv2dSTDP",
    "L4ToL23PoolingInput",
    "L56ApicalFeedbackInput",
    "L23ToL56ReciprocalBindingInput",
    "L56GatedDecisionInput",
    "SpikeTimingEligibility",
    "DelayedDopaminePlasticity",
]


class MotorToL56Current(Behavior):
    """Convert a location-vector pulse into sustained L5/6 input current."""

    def initialize(self, synapse: SynapseGroup) -> None:
        super().initialize(synapse)
        synapse.current = torch.zeros(
            1, synapse.dst.size, device=synapse.device, dtype=synapse.def_dtype
        )
        synapse.network.active_location_current = torch.zeros_like(
            synapse.current
        )

    def forward(self, synapse: SynapseGroup) -> None:
        pulse = synapse.src.spikes.to(synapse.def_dtype)
        active_rows = pulse.amax(dim=1) > 0
        if active_rows.any():
            synapse.network.active_location_current[active_rows] = (
                pulse[active_rows]
                * float(synapse.network.cfg.l56_motor_gain)
            )
        synapse.current = synapse.network.active_location_current


class SpikingConvolutionInput(Behavior):
    """Convolve only sensory spike events into the L4 dendritic current."""

    def __init__(self, kernels: torch.Tensor, *args, **kwargs) -> None:
        super().__init__(*args, kernels=kernels, **kwargs)

    def initialize(self, synapse: SynapseGroup) -> None:
        super().initialize(synapse)
        kernels = self.parameter("kernels", None, required=True).float().to(
            synapse.device
        )
        synapse.raw_kernels = kernels.clone()
        kernel_norm = synapse.raw_kernels.sum(
            dim=(1, 2, 3), keepdim=True
        ).clamp(min=1e-6)
        synapse.kernels = synapse.raw_kernels / kernel_norm
        synapse.current = torch.zeros(
            1, synapse.dst.size, device=synapse.device, dtype=synapse.def_dtype
        )

    def forward(self, synapse: SynapseGroup) -> None:
        net = synapse.network
        if net.cortical_phase != "sensory":
            synapse.current = torch.zeros_like(synapse.current)
            return
        spikes = synapse.src.spikes.to(synapse.def_dtype).reshape(
            net.batch_size, 1, net.cfg.patch_size, net.cfg.patch_size
        )
        current = F.conv2d(spikes, synapse.kernels).clamp(min=0)
        target = (net.cfg.feature_map_size, net.cfg.feature_map_size)
        if current.shape[-2:] != target:
            current = F.interpolate(
                current, size=target, mode="bilinear", align_corners=False
            )
        synapse.current = current.reshape(net.batch_size, -1)


class BatchedConv2dSTDP(Behavior):
    """Local convolutional STDP for the batched raw-image L4 pathway.

    Presynaptic sensory spikes are accumulated during the retinal phase. At
    the first L4 release step, spatial winner competition identifies one
    postsynaptic feature per receptive-field location. Only those naturally
    emitted winner spikes modify their shared convolutional kernels.
    """

    def __init__(
        self,
        *args,
        learning_rate: float = 0.05,
        a_plus: float = 1.0,
        a_minus: float = 0.1,
        homeostasis: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            learning_rate=learning_rate,
            a_plus=a_plus,
            a_minus=a_minus,
            homeostasis=homeostasis,
            **kwargs,
        )

    def initialize(self, synapse: SynapseGroup) -> None:
        super().initialize(synapse)
        self.learning_rate = float(self.parameter("learning_rate", 0.05))
        self.a_plus = float(self.parameter("a_plus", 1.0))
        self.a_minus = float(self.parameter("a_minus", 0.1))
        self.homeostasis = float(self.parameter("homeostasis", 0.05))
        synapse.stdp_pre_activity = torch.zeros(
            1,
            synapse.src.size,
            device=synapse.device,
            dtype=synapse.def_dtype,
        )
        synapse.stdp_pending = False
        synapse.l4_win_count = torch.zeros(
            int(synapse.network.n_l4_features),
            device=synapse.device,
            dtype=synapse.def_dtype,
        )
        synapse.l4_stdp_update_count = 0
        synapse.network.l4_learning_enabled = False

    def forward(self, synapse: SynapseGroup) -> None:
        net = synapse.network
        if not getattr(net, "l4_learning_enabled", False):
            return

        if net.cortical_phase == "sensory":
            spikes = synapse.src.spikes.to(synapse.def_dtype)
            if tuple(synapse.stdp_pre_activity.shape) != tuple(spikes.shape):
                synapse.stdp_pre_activity = torch.zeros_like(spikes)
            synapse.stdp_pre_activity.add_(spikes)
            synapse.stdp_pending = True
            return

        if net.cortical_phase != "l4_release" or not synapse.stdp_pending:
            return

        batch_size = int(net.batch_size)
        n_features = int(net.n_l4_features)
        n_positions = int(net.cfg.feature_map_size) ** 2
        pre = synapse.stdp_pre_activity / max(int(net.cfg.sensory_steps), 1)
        images = pre.reshape(
            batch_size,
            1,
            int(net.cfg.patch_size),
            int(net.cfg.patch_size),
        )
        kernel_size = tuple(int(value) for value in synapse.kernels.shape[-2:])
        patches = F.unfold(images, kernel_size=kernel_size)
        responses = F.conv2d(images, synapse.kernels).reshape(
            batch_size,
            n_features,
            n_positions,
        )
        usage = synapse.l4_win_count / synapse.l4_win_count.sum().clamp(min=1.0)
        scores = responses - self.homeostasis * usage[None, :, None]
        winner_index = scores.argmax(dim=1)
        winners = F.one_hot(winner_index, num_classes=n_features).permute(0, 2, 1)
        post = synapse.dst.spikes.reshape(
            batch_size,
            n_features,
            n_positions,
        )
        winners = winners.to(synapse.def_dtype) * post
        counts = winners.sum(dim=(0, 2))
        active = counts > 0
        if active.any():
            feature_patches = torch.einsum("bfp,bkp->fk", winners, patches)
            feature_patches = feature_patches / counts[:, None].clamp(min=1.0)
            weights = synapse.raw_kernels.reshape(n_features, -1)
            potentiation = feature_patches * (1.0 - weights)
            depression = (1.0 - feature_patches) * weights
            delta = self.learning_rate * (
                self.a_plus * potentiation - self.a_minus * depression
            )
            with torch.no_grad():
                weights[active] = (
                    weights[active] + delta[active]
                ).clamp(0.0, 1.0)
                norm = synapse.raw_kernels.sum(
                    dim=(1, 2, 3), keepdim=True
                ).clamp(min=1e-6)
                synapse.kernels.copy_(synapse.raw_kernels / norm)
                synapse.l4_win_count.add_(counts)
            synapse.l4_stdp_update_count += int(active.sum().item())

        synapse.stdp_pre_activity.zero_()
        synapse.stdp_pending = False


class L4ToL23PoolingInput(Behavior):
    """Pool L4 spikes into L2/3 proximal dendritic current."""

    def initialize(self, synapse: SynapseGroup) -> None:
        super().initialize(synapse)
        synapse.current = torch.zeros(
            1, synapse.dst.size, device=synapse.device, dtype=synapse.def_dtype
        )

    def forward(self, synapse: SynapseGroup) -> None:
        net = synapse.network
        if net.cortical_phase != "l4_release":
            synapse.current = torch.zeros_like(synapse.current)
            return
        spikes = synapse.src.spikes.to(synapse.def_dtype).reshape(
            net.batch_size,
            net.n_l4_features,
            net.cfg.feature_map_size,
            net.cfg.feature_map_size,
        )
        pooled = F.adaptive_avg_pool2d(
            spikes, (net.cfg.pool_grid_size, net.cfg.pool_grid_size)
        )
        synapse.current = pooled.reshape(net.batch_size, -1)


class L56ApicalFeedbackInput(Behavior):
    """Learned L5/6-to-L2/3 apical synapses driven by live L5/6 spikes."""

    def initialize(self, synapse: SynapseGroup) -> None:
        super().initialize(synapse)
        net = synapse.network
        expected = (synapse.src.size, synapse.dst.size)
        if getattr(synapse, "weights", None) is None:
            synapse.weights = torch.zeros(
                expected,
                device=synapse.device,
                dtype=synapse.def_dtype,
            )
        elif tuple(synapse.weights.shape) != expected:
            raise ValueError(
                f"apical weights must have shape {expected}, got "
                f"{tuple(synapse.weights.shape)}"
            )
        synapse.current = torch.zeros(
            1, synapse.dst.size, device=synapse.device, dtype=synapse.def_dtype
        )
        synapse.pre_trace = torch.zeros(
            1, synapse.src.size, device=synapse.device, dtype=synapse.def_dtype
        )
        synapse.post_trace = torch.zeros(
            1, synapse.dst.size, device=synapse.device, dtype=synapse.def_dtype
        )
        net.apical_learning_enabled = True

    def forward(self, synapse: SynapseGroup) -> None:
        net = synapse.network
        l56 = synapse.src.spikes.to(synapse.def_dtype)
        synapse.current = F.linear(l56, synapse.weights.T)

        if not net.apical_learning_enabled or net.cortical_phase != "decision":
            return
        l23 = synapse.dst.spikes.to(synapse.def_dtype)
        pre_decay = torch.exp(
            torch.tensor(-1.0 / 8.0, device=synapse.device)
        )
        post_decay = torch.exp(
            torch.tensor(-1.0 / 8.0, device=synapse.device)
        )
        synapse.pre_trace = pre_decay * synapse.pre_trace + l56
        synapse.post_trace = post_decay * synapse.post_trace + l23
        dw = torch.einsum("bl,bf->lf", synapse.pre_trace, l23)
        dw = dw / max(int(net.batch_size), 1)
        with torch.no_grad():
            synapse.weights.mul_(1.0 - float(net.cfg.apical_decay))
            synapse.weights.add_(float(net.cfg.apical_learning_rate) * dw)


class L23ToL56ReciprocalBindingInput(Behavior):
    """Learned L2/3-to-L5/6 synapses for feature-to-frame retrieval."""

    def initialize(self, synapse: SynapseGroup) -> None:
        super().initialize(synapse)
        expected = (synapse.dst.size, synapse.src.size)
        if getattr(synapse, "weights", None) is None:
            synapse.weights = torch.zeros(
                expected,
                device=synapse.device,
                dtype=synapse.def_dtype,
            )
        elif tuple(synapse.weights.shape) != expected:
            raise ValueError(
                f"reciprocal weights must have shape {expected}, got "
                f"{tuple(synapse.weights.shape)}"
            )
        synapse.current = torch.zeros(
            1, synapse.dst.size, device=synapse.device, dtype=synapse.def_dtype
        )
        synapse.pre_trace = torch.zeros(
            1, synapse.src.size, device=synapse.device, dtype=synapse.def_dtype
        )
        synapse.post_trace = torch.zeros(
            1, synapse.dst.size, device=synapse.device, dtype=synapse.def_dtype
        )
        synapse.update_count = 0
        synapse.network.reciprocal_learning_enabled = False
        synapse.network.reciprocal_current_enabled = True

    def forward(self, synapse: SynapseGroup) -> None:
        net = synapse.network
        l23 = synapse.src.spikes.to(synapse.def_dtype)
        synapse.current = F.linear(l23, synapse.weights)

        if (
            not getattr(net, "reciprocal_learning_enabled", False)
            or net.cortical_phase != "decision"
        ):
            return
        l56 = synapse.dst.spikes.to(synapse.def_dtype)
        trace_decay = torch.exp(
            torch.tensor(
                -1.0 / float(net.cfg.reciprocal_trace_tau),
                device=synapse.device,
            )
        )
        synapse.pre_trace = trace_decay * synapse.pre_trace + l23
        synapse.post_trace = trace_decay * synapse.post_trace + l56
        dw = torch.einsum("bl,bf->lf", l56, synapse.pre_trace)
        dw = dw / max(int(net.batch_size), 1)
        with torch.no_grad():
            synapse.weights.mul_(1.0 - float(net.cfg.reciprocal_decay))
            synapse.weights.add_(float(net.cfg.reciprocal_learning_rate) * dw)
        synapse.update_count += int(l56.sum().item())


class L56GatedDecisionInput(Behavior):
    """Coincidence-gated synapses from L2/3 and L5/6 to decision neurons."""

    def initialize(self, synapse: SynapseGroup) -> None:
        super().initialize(synapse)
        net = synapse.network
        shape = (synapse.dst.size, net.n_locations, synapse.src.size)
        generator = torch.Generator(device=synapse.device).manual_seed(
            int(net.cfg.random_state)
        )
        initial = torch.rand(
            shape,
            generator=generator,
            device=synapse.device,
            dtype=synapse.def_dtype,
        ) * 1e-3
        synapse.excitatory_weights = initial
        synapse.inhibitory_weights = torch.zeros_like(initial)
        synapse.current = torch.zeros(
            1, synapse.dst.size, device=synapse.device, dtype=synapse.def_dtype
        )

    @property
    def weights(self) -> torch.Tensor:
        raise AttributeError("Access effective weights through the synapse instance")

    def forward(self, synapse: SynapseGroup) -> None:
        net = synapse.network
        if net.cortical_phase != "decision":
            synapse.current = torch.zeros_like(synapse.current)
            net.gated_pre_spikes = torch.zeros(
                net.batch_size,
                net.n_locations,
                synapse.src.size,
                device=synapse.device,
                dtype=synapse.def_dtype,
            )
            return
        l23 = synapse.src.spikes.to(synapse.def_dtype)
        l56 = net.l56_ng.spikes.to(synapse.def_dtype)
        gated = l56.unsqueeze(2) * l23.unsqueeze(1)
        net.gated_pre_spikes = gated
        effective = synapse.excitatory_weights - synapse.inhibitory_weights
        synapse.current = (
            torch.einsum("blf,clf->bc", gated, effective)
            * float(net.cfg.decision_synaptic_gain)
        )


class SpikeTimingEligibility(Behavior):
    """Class-specific eligibility built during the original free decision."""

    def initialize(self, synapse: SynapseGroup) -> None:
        super().initialize(synapse)
        net = synapse.network
        synapse.pre_trace = torch.zeros(
            1,
            net.n_locations,
            synapse.src.size,
            device=synapse.device,
            dtype=synapse.def_dtype,
        )
        synapse.post_trace = torch.zeros(
            1, synapse.dst.size, device=synapse.device, dtype=synapse.def_dtype
        )
        # The synapse-specific field is represented in factored form to avoid
        # materializing batch x class x location x feature tensors. Their
        # product at outcome time is the same free-trial pre/post eligibility.
        synapse.eligibility = torch.zeros(
            1,
            net.n_locations,
            synapse.src.size,
            device=synapse.device,
            dtype=synapse.def_dtype,
        )
        synapse.post_eligibility = torch.zeros(
            1,
            synapse.dst.size,
            device=synapse.device,
            dtype=synapse.def_dtype,
        )

    def forward(self, synapse: SynapseGroup) -> None:
        net = synapse.network
        if (
            net.cortical_phase != "decision"
            or not getattr(net, "eligibility_enabled", True)
        ):
            return
        pre = net.gated_pre_spikes
        post = synapse.dst.spikes.to(synapse.def_dtype)
        pre_decay = torch.exp(
            torch.tensor(
                -1.0 / float(net.cfg.eligibility_tau_pre),
                device=synapse.device,
            )
        )
        post_decay = torch.exp(
            torch.tensor(
                -1.0 / float(net.cfg.eligibility_tau_post),
                device=synapse.device,
            )
        )
        eligibility_decay = torch.exp(
            torch.tensor(
                -1.0 / float(net.cfg.eligibility_tau),
                device=synapse.device,
            )
        )
        synapse.pre_trace = pre_decay * synapse.pre_trace + pre
        synapse.post_trace = post_decay * synapse.post_trace + post
        # Every decision assembly fires only according to its own LIF state.
        # Feature and postsynaptic factors are combined only when the delayed
        # outcome arrives; no target or competitor spike is imposed.
        synapse.eligibility = (
            eligibility_decay * synapse.eligibility + pre
        )
        synapse.post_eligibility = (
            eligibility_decay * synapse.post_eligibility
            + post
        )


class DelayedDopaminePlasticity(Behavior):
    """Apply outcome modulation to free-trial class eligibility traces."""

    def initialize(self, synapse: SynapseGroup) -> None:
        super().initialize(synapse)
        synapse.dopamine = torch.tensor(
            0.0, device=synapse.device, dtype=synapse.def_dtype
        )
        synapse.dopamine_abs = torch.tensor(
            0.0, device=synapse.device, dtype=synapse.def_dtype
        )
        synapse.update_count = 0

    def forward(self, synapse: SynapseGroup) -> None:
        net = synapse.network
        if net.cortical_phase != "dopamine":
            return
        pulse = net.dopamine_pulse.to(synapse.device, dtype=synapse.def_dtype)
        mask = net.dopamine_mask.to(synapse.device, dtype=synapse.def_dtype)
        if pulse.ndim != 2 or pulse.shape[1] != synapse.dst.size:
            raise ValueError(
                "dopamine_pulse must have shape (batch, decision classes)"
            )
        update = torch.zeros_like(synapse.excitatory_weights)
        learning_rate = float(net.cfg.decision_learning_rate)
        for class_index in range(synapse.dst.size):
            active = (mask > 0) & (pulse[:, class_index] != 0)
            if not active.any():
                continue
            # A local biochemical eligibility tag saturates after at least one
            # natural pre/post coincidence in the free trial. This prevents
            # tonic firing-rate differences from masquerading as stronger
            # credit while retaining strictly spike-dependent eligibility.
            selected = (
                (
                    synapse.eligibility[active]
                    > float(net.cfg.eligibility_threshold)
                ).to(synapse.def_dtype)
                * (
                    synapse.post_eligibility[
                        active, class_index, None, None
                    ] > float(net.cfg.eligibility_threshold)
                ).to(synapse.def_dtype)
            )
            norm = selected.square().sum(
                dim=(1, 2),
                keepdim=True,
            ).sqrt()
            selected = selected / norm.clamp(min=1.0)
            signed = (
                learning_rate
                * pulse[active, class_index, None, None]
                * selected
            )
            update[class_index] = signed.sum(dim=0)
        with torch.no_grad():
            turnover = max(
                0.0,
                1.0 - float(net.cfg.decision_weight_decay),
            )
            synapse.excitatory_weights.mul_(turnover)
            synapse.inhibitory_weights.mul_(turnover)
            positive = update.clamp(min=0)
            negative = (-update).clamp(min=0)
            synapse.excitatory_weights.add_(positive)
            synapse.inhibitory_weights.add_(negative)
            synapse.excitatory_weights.clamp_(0.0, float(net.cfg.weight_max))
            synapse.inhibitory_weights.clamp_(0.0, float(net.cfg.weight_max))
        synapse.dopamine = (
            synapse.dopamine
            - synapse.dopamine / float(net.cfg.dopamine_tau)
            + pulse.mean()
        )
        synapse.dopamine_abs = pulse.abs().mean()
        synapse.update_count += int(mask.sum().item())
