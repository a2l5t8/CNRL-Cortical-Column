"""One shared CoNeX graph with canonical training and prediction APIs."""

from dataclasses import asdict, dataclass

import torch
from conex import Neocortex, PreSpikeCatcher, SynapseInit, WeightInitializer
from pymonntorch import SynapseGroup

from .behavior.network import RewardPulse
from .behavior.synapse import (
    DenseDendriticInput,
    HeldSpikeTransmission,
    PairSTDP,
    PooledBurstInput,
)
from .DecisionLayer import DecisionLayer
from .L4 import L4, L4Config, fixation_geometry
from .L23 import L23
from .L56 import L56
from .simulation import activate, set_batch, simulate


@dataclass
class Config:
    classes: int = 2
    maps: int = 25
    kernel_size: int = 11
    pool: int = 5
    winners: int = 5
    sensory_steps: int = 32
    decision_steps: int = 64
    learning_rate: float = 0.003
    noise: float = 0.003
    decision_gain: float | None = None
    seed: int = 42
    device: str = "cuda"
    grid: int = 3
    reference_frames: bool = True


class Column:
    """Train and predict with a shared CoNeX/PymoNNtorch network.

    ``train`` performs learning and returns per-epoch metrics. ``predict``
    returns CPU class indices without modifying learned state. Both methods
    initialize the network automatically if it has not been initialized yet.
    """

    def __init__(self, config=None):
        self.config = config or Config()
        c = self.config
        if min(c.classes, c.maps, c.sensory_steps, c.decision_steps) < 1:
            raise ValueError("Population sizes and simulation windows must be positive")
        if not 1 <= c.winners <= c.maps * c.pool**2:
            raise ValueError("Winner count must fit the pooled feature population")
        patch_size, self.centers = fixation_geometry(c.grid)
        if not 1 <= c.kernel_size <= patch_size or c.pool < 1:
            raise ValueError(
                "Kernel must fit the glimpse and pooling side must be positive"
            )

        self.net = Neocortex(device=c.device, behavior={10: RewardPulse()})
        self.net.fixations = len(self.centers)
        self.net.batch_size = 1
        self.net.binding = False
        self.net.drive = None
        kernels = torch.rand(
            c.maps,
            1,
            c.kernel_size,
            c.kernel_size,
            generator=torch.Generator().manual_seed(c.seed),
        )
        self.l4 = L4(
            self.net,
            kernels,
            L4Config(patch_size=patch_size, pool=c.pool, steps=c.sensory_steps),
        )
        self.l23 = L23(self.net, c.maps * c.pool**2, c.winners)
        self.l56 = L56(self.net, c.classes, c.grid) if c.reference_frames else None
        self._connect_context()
        self.decision = DecisionLayer(
            self.net,
            c.maps * c.pool**2,
            c.classes,
            c.winners,
            lr=c.learning_rate,
            noise=c.noise,
            gain=c.decision_gain,
            seed=c.seed,
            grid=c.grid,
            position=c.reference_frames,
        )
        self._connect_readout()
        self.stages = {"l4", "bindings", "decision"}
        self.history = []
        self.validation_predictions = None
        self._shuffle_generator = torch.Generator().manual_seed(c.seed + 10000)

    def _connect_context(self):
        self._context_objects = list(self.l23.objects)
        feedforward = SynapseGroup(
            net=self.net,
            src=self.l4.pool,
            dst=self.l23.neurons,
            tag="Proximal",
            behavior={2: SynapseInit(), 180: PooledBurstInput(source="drive")},
        )
        self._context_objects.append(feedforward)
        if self.l56 is None:
            return
        self._context_objects.extend(self.l56.objects)
        for source, layer, scale in (
            (self.l56.neurons, self.l23, self.config.winners),
            (self.l23.neurons, self.l56, 1),
        ):
            layer.apical = SynapseGroup(
                net=self.net,
                src=source,
                dst=layer.neurons,
                tag="Apical",
                behavior={
                    2: SynapseInit(),
                    3: WeightInitializer(
                        weights=torch.zeros(
                            source.size, layer.neurons.size, device=self.net.device
                        )
                    ),
                    180: DenseDendriticInput(),
                    420: PreSpikeCatcher(),
                    500: PairSTDP(scale=scale),
                },
            )
            self._context_objects.append(layer.apical)

    def _connect_readout(self):
        connections = [
            (self.l23.neurons, self.decision.features, HeldSpikeTransmission())
        ]
        if self.l56 is not None:
            connections.append(
                (
                    self.l56.neurons,
                    self.decision.locations,
                    HeldSpikeTransmission(
                        locations=self.l56.object_locations,
                        classes=self.config.classes,
                        frame_size=self.l56.frame_size,
                    ),
                )
            )
        for source, destination, transmission in connections:
            bridge = SynapseGroup(
                net=self.net,
                src=source,
                dst=destination,
                tag="Proximal",
                behavior={2: SynapseInit(), 90: transmission},
            )
            self.decision.objects.append(bridge)

    def initialize(self):
        """Initialize once; explicit ``net.initialize()`` is also supported."""
        if not hasattr(self.l4.conv, "raw_weights"):
            self.net.initialize(info=False, warnings=False)
        return self

    def plasticity(self, stages=("l4", "bindings", "decision")):
        """Select the pathways that learn during subsequent training calls."""
        stages = set(stages)
        if stages - {"l4", "bindings", "decision"}:
            raise ValueError("Unknown plasticity stage")
        self.stages = stages
        return self

    @staticmethod
    def _images(images):
        if images.ndim == 2:
            images = images.unsqueeze(0)
        if images.ndim != 3 or tuple(images.shape[1:]) != (80, 80):
            raise ValueError("Expected rate images with shape (N, 80, 80)")
        return images

    def _labels(self, labels, count):
        labels = torch.as_tensor(labels, dtype=torch.long).cpu()
        if labels.ndim != 1 or len(labels) != count:
            raise ValueError("Provide one class index per image")
        if ((labels < 0) | (labels >= self.config.classes)).any():
            raise ValueError("Class index is outside the configured class range")
        return labels

    @torch.no_grad()
    def train(
        self,
        images,
        labels,
        *,
        epochs=1,
        batch_size=32,
        shuffle=True,
        seed=None,
        validation_data=None,
        callback=None,
    ):
        """Learn from rate images and return this call's epoch metrics.

        Labels supply the training-only frame cue and external scalar reward.
        Optional ``callback(column, metrics)`` runs after each completed epoch.
        The selected plasticity stages remain enabled for every training batch.
        """
        images = self._images(images)
        labels = self._labels(labels, len(images))
        if not len(images) or epochs < 1 or batch_size < 1:
            raise ValueError(
                "Training requires images, positive epochs and positive batch size"
            )
        if validation_data is not None:
            validation_images = self._images(validation_data[0])
            validation_labels = self._labels(validation_data[1], len(validation_images))
            if not len(validation_images):
                raise ValueError("Validation data must not be empty")
        if seed is not None:
            self._shuffle_generator.manual_seed(seed)
        self.initialize()
        results = []
        for _ in range(epochs):
            order = (
                torch.randperm(len(images), generator=self._shuffle_generator)
                if shuffle
                else torch.arange(len(images))
            )
            correct = 0
            for indices in order.split(batch_size):
                predictions = self._present(
                    images[indices], labels[indices], training=True
                )
                correct += int((predictions.cpu() == labels[indices]).sum())
            metrics = {
                "epoch": len(self.history) + 1,
                "training_accuracy": correct / len(images),
            }
            self.validation_predictions = None
            if validation_data is not None:
                self.validation_predictions = self.predict(
                    validation_images, batch_size=batch_size
                )
                metrics["validation_accuracy"] = int(
                    (self.validation_predictions == validation_labels).sum()
                ) / len(validation_labels)
            self.history.append(metrics)
            results.append(dict(metrics))
            if callback is not None:
                callback(self, dict(metrics))
        return results

    @torch.no_grad()
    def predict(self, images, *, batch_size=128):
        """Return CPU class indices with plasticity and exploration disabled."""
        images = self._images(images)
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        self.initialize()
        predictions = [
            self._present(batch).cpu()
            for batch in images.split(batch_size)
            if len(batch)
        ]
        return (
            torch.cat(predictions) if predictions else torch.empty(0, dtype=torch.long)
        )

    def _present(self, images, labels=None, training=False):
        drives = (
            25
            * self.l4.drive(
                images, self.centers, learn=training and "l4" in self.stages
            )
        ).round()
        if training and self.l56 is not None and "bindings" in self.stages:
            self._learn_bindings(drives, labels)
        codes = self._feature_codes(drives)
        self.decision.trial(
            codes.float() * 20,
            [self.config.decision_steps] * len(self.centers),
            learning=training and "decision" in self.stages,
        )
        predictions = self.decision.release()
        if training and "decision" in self.stages:
            self.decision.feedback((predictions == labels.to(self.net.device)).float())
        return predictions

    def _learn_bindings(self, drives, labels):
        activate(self.net, self._context_objects)
        set_batch(self.net, len(drives), self._context_objects)
        self.net.binding = True
        self.net.drive = None
        labels = labels.to(self.net.device)
        features = self.config.maps * self.config.pool**2
        for slot, location in enumerate(self.l56.object_locations):
            values, indices = drives[:, slot].topk(
                self.config.winners, dim=1, sorted=False
            )
            feature_spikes = torch.zeros(
                len(drives), features, device=self.net.device
            ).scatter_(1, indices, (values >= 12.5).float())
            location_spikes = torch.nn.functional.one_hot(
                labels * self.l56.frame_size + location,
                self.config.classes * self.l56.frame_size,
            ).float()
            for synapse in (self.l23.apical, self.l56.apical):
                synapse.local_pre.zero_()
                synapse.local_post.zero_()
            self.l56.neurons.binding_spikes = location_spikes
            for pulse in (
                torch.zeros_like(feature_spikes),
                feature_spikes,
                torch.zeros_like(feature_spikes),
            ):
                self.l23.neurons.binding_spikes = pulse
                simulate(self.net)
        self.net.binding = False

    def _feature_codes(self, drives):
        activate(self.net, self._context_objects)
        batch = len(drives)
        set_batch(self.net, batch, self._context_objects)
        self.net.binding = False
        self.net.drive = None
        if self.l56 is not None:
            self.l56.cue.external = torch.ones(
                batch, self.config.classes, dtype=torch.bool, device=self.net.device
            )
        result = []
        for slot in range(len(self.centers)):
            self.net.slot = slot
            if self.l56 is not None:
                self.l56.motor.external = torch.zeros(
                    batch, len(self.centers), dtype=torch.bool, device=self.net.device
                )
                self.l56.motor.external[:, slot] = True
            simulate(self.net, 2)
            self.net.drive = (
                drives[:, slot].flatten().to(self.net.device, torch.float32)
            )
            self.net.capture_context = True
            simulate(self.net)
            self.net.capture_context = False
            result.append(self.l23.neurons.spikes.reshape(batch, -1).clone())
            simulate(self.net, 31)
            self.net.drive = None
            simulate(self.net)
        return torch.stack(result, 1)

    def state_dict(self):
        """Serialize learned state without pickling framework objects."""
        self.initialize()
        return {
            "format": "framework-column-v1",
            "config": asdict(self.config),
            "kernels": self.l4.kernels,
            "wins": self.l4.conv.wins.cpu(),
            "context": None
            if self.l56 is None
            else {
                "n_objects": self.config.classes,
                "n_features": self.config.maps * self.config.pool**2,
                "k": self.config.winners,
                "raw_forward": self.l23.apical.raw.cpu(),
                "raw_reverse": self.l56.apical.raw.cpu(),
                "forward": self.l23.apical.weights.cpu(),
                "reverse": self.l56.apical.weights.cpu(),
            },
            "decision": self.decision.state_dict(),
            "history": self.history,
            "plasticity": sorted(self.stages),
            "shuffle_generator": self._shuffle_generator.get_state(),
        }

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, device="cuda"):
        """Restore new checkpoints and the existing framework-column-v1 files."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format") != "framework-column-v1":
            raise ValueError("Not a framework column checkpoint")
        model = cls(Config(**dict(state["config"], device=device)))
        model.initialize()
        model.l4.conv.raw_weights.copy_(state["kernels"].to(device))
        model.l4.stdp.scale(model.l4.conv)
        model.l4.conv.wins.copy_(state["wins"].to(device))
        if model.l56 is not None:
            for synapse, raw_key, weight_key in (
                (model.l23.apical, "raw_forward", "forward"),
                (model.l56.apical, "raw_reverse", "reverse"),
            ):
                synapse.raw = state["context"][raw_key].to(device).clone()
                synapse.weights = state["context"][weight_key].to(device).clone()
        model.decision.w = state["decision"]["w"].to(device).clone()
        model.net.baseline = state["decision"]["reward_baseline"]
        model.decision.generator.set_state(state["decision"]["generator"].cpu())
        model.history = state.get("history", [])
        model.plasticity(state.get("plasticity", ("l4", "bindings", "decision")))
        if "shuffle_generator" in state:
            model._shuffle_generator.set_state(state["shuffle_generator"])
        return model
