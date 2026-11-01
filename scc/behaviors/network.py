"""Network behaviors the cortical column adds to CoNeX."""
import torch
from conex.behaviors.network import Payoff

__all__ = ["MarginPayoff"]


class MarginPayoff(Payoff):
    """Outcome signal of a trial: payoff 1 when the decision was not secure,
    i.e. the attended target's spike count does not exceed that of every rival
    by at least ``margin``; 0 otherwise and whenever no outcome is due.

    The payoff is one scalar per trial (a global signal); CoNeX's Dopamine
    turns it into the dopamine concentration read by GatedRSTDP.

    Note: set ``network.outcome_due`` on the step the outcome arrives.

    Args:
        margin (float): Required lead of the target over the strongest rival.
        decision (SynapseGroup): Decision synapses (PostTrace counts, apical teaching).
    """

    def __init__(self, margin, decision, *args, **kwargs):
        super().__init__(*args, margin=margin, decision=decision, **kwargs)

    def initialize(self, network):
        super().initialize(network)
        self.margin = self.parameter("margin", None, required=True)
        self.decision = self.parameter("decision", None, required=True)
        network.outcome_due = False

    def forward(self, network):
        if not network.outcome_due:
            network.payoff = 0.0
            return
        syn = self.decision
        b, C = network.batch_size, syn.weights.shape[0]
        anchor = syn.anchor[:, None, None].expand(-1, C, 1)
        counts = syn.post_trace.view(b, C, -1).gather(2, anchor).squeeze(2)
        taught = syn.dst.I_apical.view(b, C, -1).gather(2, anchor).squeeze(2) > 0
        target = counts.masked_fill(~taught, -torch.inf).amax(1)
        rival = counts.masked_fill(taught, -torch.inf).amax(1)
        network.payoff = (target - rival < self.margin).to(counts.dtype)
