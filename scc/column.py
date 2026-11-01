"""Cortical column with object-specific reference frames (Thousand Brains layer 6).

Populations (all spiking, built from CoNeX behaviors):

* L2/3: feature neurons. Proximal input is the feed-forward drive of the
  sensory pathway; apical input from the location units primes the features
  the active frames predict. k-WTA keeps the k most depolarised neurons.
* hypotheses H[c, d]: object c anchored at offset d. They integrate the
  log-likelihood evidence of reciprocal L2/3 synapses and compete by
  winner-take-all with a margin, so ambiguity narrows across saccades.
* locations: one reference frame of location units per object. Hypothesis
  (c, d) and the motor position p activate location l = p - d of frame c
  (path integration).
* decision units (c, d): integrate L2/3 spikes through the dopamine-trained
  synapses of the object location their hypothesis occupies.
* motor, cue and teacher inputs: the sensor position; a proximal cue that
  holds hypotheses active (the attended object during learning, or a known
  position); an apical teaching signal marking the target decision unit.
"""
import contextlib
import io
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import torch
from conex import (KWTA, LIF, Dopamine, Fire, LocationSetter, Neocortex, NeuronAxon, One2OneDendriticInput,
                   PostSpikeCatcher, PostTrace, PreSpikeCatcher, SensorySetter, SimpleDendriteComputation,
                   SimpleDendriteStructure, SynapseInit, WeightInitializer, prioritize_behaviors)
from conex.nn.structure.layer import Layer
from pymonntorch import NeuronGroup, SynapseGroup

from scc.behaviors import (IF, DenseDendriticInput, FrameDendriticInput, FrameGatedDendriticInput, GatedRSTDP,
                           MarginPayoff, MarginWTA, Outstar, PathIntegrationDendriticInput, ReplayDendriticInput)
from scc.utils import gate, set_batch, simulate

LATTICE = [(r, c) for r in range(5) for c in range(5)]  # 5x5 fixation lattice (20-px steps on the canvas)
OFFSETS = [(dy, dx) for dy in range(3) for dx in range(3)]  # candidate object placements
CENTRE = (1, 1)  # placement of a centred object
INNER = [6, 7, 8, 11, 12, 13, 16, 17, 18]  # lattice positions on a centred object, raster order
CUE = 1e6  # proximal cue current; dominates any accumulated evidence


@dataclass
class ColumnConfig:
    k: int = 16  # L2/3 winners per fixation
    l23_threshold: float = 12.5  # pooled L4 spikes needed to compete (half a spike per cell of a 5x5 pool)
    apical_provocativeness: float = 0.0  # apical priming depolarises L2/3 up to this fraction of threshold
    decision_steps: int = 8
    decision_gain: float = 0.3
    tonic: float = 0.5
    lr: float = 0.05
    margin: float = 6.0  # required spike-count lead of the target
    w_max: float = 5.0
    elimination_margin: float = 10.0  # nats a hypothesis may trail the leader and stay active
    smoothing: float = 0.5  # pseudo-count of the binding statistics
    background_exposures: int = 50
    seed: int = 42


class ReferenceFrames:
    """Geometry of C object frames: 5x5 object-centred locations (-1..3; the
    object covers 0..2) plus a far-background location, anchors ``offsets``
    and fixation ``lattice``."""

    LO, HI = -1, 3

    def __init__(self, n_objects: int, lattice, offsets, device):
        self.lattice, self.offsets = [tuple(p) for p in lattice], [tuple(d) for d in offsets]
        side = self.HI - self.LO + 1
        self.background = side * side
        self.C, self.D, self.L = n_objects, len(self.offsets), self.background + 1
        self.table = torch.tensor([[self.index(r - dy, c - dx) for dy, dx in self.offsets]
                                   for r, c in self.lattice], device=device)  # (P, D)
        self.object_locations = [self.index(ly, lx) for ly in range(3) for lx in range(3)]
        self.slot = torch.full((self.L,), -1, dtype=torch.long, device=device)
        self.slot[self.object_locations] = torch.arange(9, device=device)

    def index(self, ly: int, lx: int) -> int:
        inside = self.LO <= ly <= self.HI and self.LO <= lx <= self.HI
        return (ly - self.LO) * (self.HI - self.LO + 1) + lx - self.LO if inside else self.background

    def footprint(self, d: int) -> List[int]:
        """Lattice positions on the object under anchor d, object raster order."""
        dy, dx = self.offsets[d]
        return [self.lattice.index((ly + dy, lx + dx)) for ly in range(3) for lx in range(3)
                if (ly + dy, lx + dx) in self.lattice]


def _input(net, size: int, setter, tag: str):
    """An external input population (spikes set from outside) in its own CoNeX Layer."""
    ng = NeuronGroup(net=net, size=size, tag=tag, behavior=prioritize_behaviors([setter, NeuronAxon()]))
    return ng, Layer(net=net, neurongroups=[ng], tag=tag)


def _neurons(net, size: int, tag: str, *soma, provocativeness=None):
    """A population with proximal and apical dendrites, the given soma behaviors and an axon."""
    return NeuronGroup(net=net, size=size, tag=tag, behavior=prioritize_behaviors([
        SimpleDendriteStructure(Apical_max_delay=1, apical_min_delay=0),
        SimpleDendriteComputation(apical_provocativeness=provocativeness), *soma, NeuronAxon()]))


def _synapse(net, src, dst, tag: str, *behaviors, weights=None):
    """A synapse group catching presynaptic spikes; ``tag`` selects the dendrite (Proximal / Apical)."""
    init = [] if weights is None else [WeightInitializer(weights=weights)]
    return SynapseGroup(net=net, src=src, dst=dst, tag=tag, behavior=prioritize_behaviors(
        [SynapseInit(), *init, PreSpikeCatcher(), *behaviors]))


class CorticalColumn:
    def __init__(self, n_objects: int, n_features: int, cfg: ColumnConfig = None, offsets=OFFSETS,
                 lattice=LATTICE, device="cpu"):
        self.cfg = cfg or ColumnConfig()
        self.device = device
        self.F = n_features
        self.frames = ReferenceFrames(n_objects, lattice, offsets, device)
        self.net = Neocortex(dt=1, device=device)
        self.net.batch_size = 1

        self._build_inputs()
        self._build_l23()
        self._build_l56()
        self._build_decision()
        self._connect_l23()
        self._connect_l56()
        self._connect_decision()
        self._build_outcome()
        with contextlib.redirect_stdout(io.StringIO()):
            self.net.initialize(info=False, warnings=False)
        self.evidence.bias = torch.zeros(self.frames.C, self.frames.L, device=device)
        self.background(self.cfg.background_exposures)

    # ------------------------------------------------------------ populations
    def _build_inputs(self) -> None:
        """Motor position, proximal cue to the hypotheses, apical teacher of the decisions,
        and L4, the source of the replayed feed-forward drive."""
        f, net = self.frames, self.net
        self.motor, self.motor_layer = _input(net, len(f.lattice), LocationSetter(), "Motor")
        self.cue, self.cue_layer = _input(net, f.C * f.D, SensorySetter(), "Cue")
        self.teacher, self.teacher_layer = _input(net, f.C * f.D, SensorySetter(), "Teacher")
        self.l4 = NeuronGroup(net=net, size=self.F, tag="L4", behavior={})

    def _build_l23(self) -> None:
        """Memoryless LIF feature neurons with apical priming and k-WTA."""
        cfg = self.cfg
        self.l23 = _neurons(self.net, self.F, "L23",
                            LIF(R=1, tau=1, threshold=cfg.l23_threshold, v_reset=0, v_rest=0),
                            KWTA(k=cfg.k, dimension=1), Fire(), provocativeness=cfg.apical_provocativeness)

    def _build_l56(self) -> None:
        """Frame hypotheses (evidence integrators competing by margin WTA) and location units."""
        f = self.frames
        self.wta = MarginWTA(margin=self.cfg.elimination_margin)
        self.hyp = _neurons(self.net, f.C * f.D, "Hypotheses", IF(), self.wta)
        self.loc = _neurons(self.net, f.C * f.L, "Locations",
                            LIF(R=1, tau=1, threshold=0.5, v_reset=0, v_rest=0), Fire())

    def _build_decision(self) -> None:
        """One IF decision unit per (object, anchor) hypothesis."""
        self.decision = _neurons(self.net, self.frames.C * self.frames.D, "Decision", IF(), Fire())

    # ------------------------------------------------------------ connections
    def _scalar(self, value: float) -> torch.Tensor:
        return torch.tensor(float(value), device=self.device)

    def _connect_l23(self) -> None:
        """Feed-forward drive (proximal) and frame priming (apical, learned by outstar)."""
        f, net = self.frames, self.net
        SynapseGroup(net=net, src=self.l4, dst=self.l23, tag="Proximal",
                     behavior=prioritize_behaviors([SynapseInit(), ReplayDendriticInput(source="drive")]))
        self.outstar = Outstar()
        self.prime = _synapse(net, self.loc, self.l23, "Apical", DenseDendriticInput(), PostSpikeCatcher(),
                              self.outstar, weights=torch.zeros(f.C * f.L, self.F, device=self.device))

    def _connect_l56(self) -> None:
        """Reciprocal evidence and cue onto the hypotheses; path integration onto the locations."""
        f, net = self.frames, self.net
        self.sense = FrameDendriticInput(frames=f, motor=self.motor)
        self.evidence = _synapse(net, self.l23, self.hyp, "Proximal", self.sense,
                                 weights=torch.zeros(f.C, f.L, self.F, device=self.device))
        _synapse(net, self.cue, self.hyp, "Proximal", One2OneDendriticInput(), weights=self._scalar(CUE))
        _synapse(net, self.hyp, self.loc, "Proximal", PathIntegrationDendriticInput(frames=f, motor=self.motor))

    def _connect_decision(self) -> None:
        """Frame-gated, three-factor-plastic L2/3 synapses; tonic drive from the hypothesis;
        apical teaching input."""
        cfg, f, net = self.cfg, self.frames, self.net
        g = torch.Generator(device=self.device).manual_seed(cfg.seed)
        w0 = torch.rand(f.C, 9, self.F, generator=g, device=self.device) * 1e-3
        self.gated = FrameGatedDendriticInput(frames=f, motor=self.motor, hypotheses=self.hyp,
                                              current_coef=cfg.decision_gain)
        self.rstdp = GatedRSTDP(lr=cfg.lr, w_max=cfg.w_max)
        self.synapses = _synapse(net, self.l23, self.decision, "Proximal", PostSpikeCatcher(),
                                 PostTrace(tau_s=float("inf")), self.gated, self.rstdp, weights=w0)
        self.tonic = One2OneDendriticInput()
        self.tonic_sg = _synapse(net, self.hyp, self.decision, "Proximal", self.tonic,
                                 weights=self._scalar(cfg.tonic))
        _synapse(net, self.teacher, self.decision, "Apical", One2OneDendriticInput(), weights=self._scalar(1.0))

    def _build_outcome(self) -> None:
        """Global outcome signal: payoff of an insecure decision -> dopamine."""
        self.net.add_behaviors(prioritize_behaviors([MarginPayoff(margin=self.cfg.margin, decision=self.synapses),
                                                     Dopamine(tau_dopamine=1.0)]))

    # ------------------------------------------------------------ bindings
    @property
    def binding(self) -> torch.Tensor:
        """Learned P(feature | object, location), (C, L, F)."""
        return self.prime.weights.view(self.frames.C, self.frames.L, self.F)

    def refresh(self) -> None:
        """Reciprocal efficacies: log-odds of the smoothed binding statistics."""
        a = self.cfg.smoothing
        n = self.prime.n.view(self.frames.C, self.frames.L, 1)
        p = (self.binding * n + a) / (n + 2 * a)
        self.evidence.weights = torch.log(p) - torch.log1p(-p)
        self.evidence.bias = torch.log1p(-p).sum(2)

    def background(self, exposures: int) -> None:
        """Every frame's far-background location has seen ``exposures`` blank fixations."""
        rows = torch.arange(self.frames.C, device=self.device) * self.frames.L + self.frames.background
        self.prime.n[rows] = float(exposures)
        self.prime.weights[rows] = 0.0
        self.refresh()

    # ----------------------------------------------------------- scheduling
    def _reset(self, b: int) -> None:
        set_batch(self.net, b)
        self.net.drive = None
        self.motor_layer.loc = self.cue_layer.x = self.teacher_layer.x = None

    def _fixate(self, p: torch.Tensor, drive: torch.Tensor, *, decide=True, learn=False) -> Dict[str, torch.Tensor]:
        """One fixation: move, let the frames prime L2/3, sense, decide, update evidence."""
        b = len(p)
        motor = torch.zeros(b, len(self.frames.lattice), dtype=torch.bool, device=self.device)
        motor[torch.arange(b), p.to(self.device)] = True
        self.motor_layer.loc = motor.view(-1)
        self._quiet()
        simulate(self.net, 2)  # motor -> locations -> apical priming of L2/3
        rec = {"active": self.hyp.spikes.view(b, -1).clone(), "location": self.loc.spikes.view(b, -1).clone()}
        self.net.drive = drive.reshape(-1).to(self.device, torch.float32)
        simulate(self.net)
        rec["prime"] = self.prime.I.view(b, -1).clone()
        rec["code"] = self.l23.spikes.view(b, -1).clone()
        gate(self.synapses, self.gated, decide)
        gate(self.tonic_sg, self.tonic, decide)
        simulate(self.net, self.cfg.decision_steps - 2)
        gate(self.prime, self.outstar, learn)
        simulate(self.net)
        gate(self.prime, self.outstar, False)
        self.net.drive = None
        gate(self.evidence, self.sense, True)
        simulate(self.net)  # the last code reaches the hypotheses
        rec["evidence"] = self.hyp.v.view(b, -1).clone()
        rec["decision"] = self.synapses.post_trace.view(b, -1).clone()
        return rec

    def _quiet(self) -> None:
        """Silence decision, evidence and binding plasticity between fixations."""
        for obj, behavior in ((self.synapses, self.gated), (self.tonic_sg, self.tonic),
                              (self.evidence, self.sense), (self.prime, self.outstar)):
            gate(obj, behavior, False)

    # ------------------------------------------------------------- learning
    @torch.no_grad()
    def learn_bindings(self, drives: torch.Tensor, labels: torch.Tensor, positions: Sequence[int],
                       anchor=CENTRE, batch_size: int = 1024) -> "CorticalColumn":
        """Bindings learned while a proximal cue holds the object's frame active."""
        d = self.frames.offsets.index(tuple(anchor))
        for s in range(0, len(drives), batch_size):
            x, y = drives[s:s + batch_size], labels[s:s + batch_size].to(self.device)
            b = len(x)
            self._reset(b)
            cue = torch.zeros(b, self.frames.C, self.frames.D, dtype=torch.bool, device=self.device)
            cue[torch.arange(b), y, d] = True
            self.cue_layer.x = cue.view(-1)
            for t, p in enumerate(positions):
                self._fixate(torch.full((b,), p), x[:, t], decide=False, learn=True)
        self.refresh()
        return self

    @torch.no_grad()
    def learn_decisions(self, drives: torch.Tensor, labels: torch.Tensor, positions: Sequence[int],
                        anchor=CENTRE, epochs: int = 1, batch_size: int = 1024,
                        seed: Optional[int] = None) -> List[Dict[str, float]]:
        """Three-factor learning of the decision synapses in free trials at a known
        position: every object's hypothesis at ``anchor`` is cued, the target's
        decision unit receives apical teaching, and the outcome signal arrives
        after the last fixation."""
        d = self.frames.offsets.index(tuple(anchor))
        gen = torch.Generator().manual_seed(self.cfg.seed if seed is None else seed)
        history = []
        for epoch in range(epochs):
            order = torch.randperm(len(drives), generator=gen)
            correct = updated = 0
            for s in range(0, len(drives), batch_size):
                idx = order[s:s + batch_size]
                x, y = drives[idx], labels[idx].to(self.device)
                b = len(x)
                self._reset(b)
                cue = torch.zeros(b, self.frames.C, self.frames.D, dtype=torch.bool, device=self.device)
                cue[:, :, d] = True
                teach = torch.zeros_like(cue)
                teach[torch.arange(b), y, d] = True
                self.cue_layer.x, self.teacher_layer.x = cue.view(-1), teach.view(-1)
                for t, p in enumerate(positions):
                    rec = self._fixate(torch.full((b,), p), x[:, t])
                counts = rec["decision"].view(b, self.frames.C, -1)[:, :, d]
                correct += int((counts.argmax(1) == y).sum())
                self._quiet()
                self.net.outcome_due = True
                simulate(self.net)
                self.net.outcome_due = False
                updated += int(self.net.payoff.sum())
            history.append({"epoch": epoch + 1, "policy_accuracy": correct / len(drives),
                            "update_fraction": updated / len(drives)})
        return history

    # ------------------------------------------------------------ inference
    def _goal(self, visited: torch.Tensor, fallback: Sequence[int], evidence: torch.Tensor) -> torch.Tensor:
        """Model-based saccade: the next unvisited location on the leading
        hypothesis' object, then the fallback order."""
        lead = evidence.view(len(visited), self.frames.C, -1).amax(1).argmax(1).tolist()
        choice = []
        for i, d in enumerate(lead):
            choice.append(next(p for p in self.frames.footprint(d) + list(fallback) if not visited[i, p]))
        return torch.tensor(choice)

    @torch.no_grad()
    def infer(self, drives: torch.Tensor, positions: Sequence[int], *, policy: str = "fixed",
              n_fixations: Optional[int] = None, start: int = 12, eliminate: bool = True,
              known_anchor=None, batch_size: int = 512) -> Dict[str, torch.Tensor]:
        """Free inference over a saccade sequence, recorded after every fixation.

        ``drives`` holds the drive of every fixation in ``positions`` order
        (policy "fixed") or of every lattice position (policy "goal": the first
        fixation is ``start``, later ones follow the leading hypothesis, with
        ``positions`` as fallback order). ``known_anchor`` cues every object's
        hypothesis at that anchor (position known), as in the original column.
        """
        f = self.frames
        T = len(positions) if n_fixations is None else n_fixations
        self.wta.margin = self.cfg.elimination_margin if eliminate else float("inf")
        keys = ("active", "location", "prime", "code", "evidence", "decision", "position")
        out = {k: [] for k in keys}
        for s in range(0, len(drives), batch_size):
            x = drives[s:s + batch_size]
            b = len(x)
            self._reset(b)
            if known_anchor is not None:
                cue = torch.zeros(b, f.C, f.D, dtype=torch.bool, device=self.device)
                cue[:, :, f.offsets.index(tuple(known_anchor))] = True
                self.cue_layer.x = cue.view(-1)
            visited = torch.zeros(b, len(f.lattice), dtype=torch.bool)
            rec, evidence = {k: [] for k in keys}, None
            for t in range(T):
                if policy == "fixed":
                    p, drive = torch.full((b,), positions[t]), x[:, t]
                else:
                    p = torch.full((b,), start) if t == 0 else self._goal(visited, positions, evidence)
                    drive = x[torch.arange(b), p]
                visited[torch.arange(b), p] = True
                r = self._fixate(p, drive)
                r["position"] = p
                evidence = r["evidence"]
                for k in keys:
                    rec[k].append(r[k].cpu())
            for k in keys:
                out[k].append(torch.stack(rec[k], 1))
        self.wta.margin = self.cfg.elimination_margin
        res = {k: torch.cat(v) for k, v in out.items()}
        N = len(drives)
        for k in ("active", "evidence", "decision"):
            res[k] = res[k].view(N, T, f.C, f.D)
        res["location"] = res["location"].view(N, T, f.C, f.L)
        res["prediction"], res["anchor"] = self.readout(res["evidence"], res["decision"])
        return res

    @staticmethod
    def readout(evidence: torch.Tensor, decision: torch.Tensor):
        """Where: the anchor of the leading hypothesis. What: decision spike
        counts of every object at that anchor (N, T, C)."""
        anchor = evidence.amax(2).argmax(2)
        counts = decision.gather(3, anchor[:, :, None, None].expand(-1, -1, decision.shape[2], 1)).squeeze(3)
        return counts, anchor

    # ----------------------------------------------------------- persistence
    @property
    def decision_weights(self) -> torch.Tensor:
        return self.synapses.weights

    def state_dict(self) -> Dict[str, object]:
        return {"cfg": asdict(self.cfg), "n_objects": self.frames.C, "n_features": self.F,
                "offsets": self.frames.offsets, "lattice": self.frames.lattice,
                "binding": self.prime.weights.cpu(), "binding_counts": self.prime.n.cpu(),
                "w_exc": self.synapses.w_exc.cpu(), "w_inh": self.synapses.w_inh.cpu()}

    def load_state_dict(self, state: Dict[str, object], bindings: bool = True) -> "CorticalColumn":
        self.synapses.w_exc = state["w_exc"].to(self.device)
        self.synapses.w_inh = state["w_inh"].to(self.device)
        self.synapses.weights = self.synapses.w_exc - self.synapses.w_inh
        if bindings and state["offsets"] == self.frames.offsets:
            self.prime.weights = state["binding"].to(self.device)
            self.prime.n = state["binding_counts"].to(self.device)
            self.refresh()
        return self

    @classmethod
    def load(cls, state: Dict[str, object], device="cpu", **overrides) -> "CorticalColumn":
        cfg = ColumnConfig(**{**state["cfg"], **overrides})
        model = cls(state["n_objects"], state["n_features"], cfg, state["offsets"], state["lattice"], device)
        return model.load_state_dict(state)
