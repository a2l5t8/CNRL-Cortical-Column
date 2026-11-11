"""Network-wide reward and dopamine signaling."""

from pymonntorch import Behavior


class RewardPulse(Behavior):
    def initialize(self, network):
        if not hasattr(network, "baseline"):
            network.baseline = 0.0

    def forward(self, network):
        if network.phase == "feedback":
            network.dopamine = network.reward - network.baseline
            network.baseline += 0.01 * (float(network.reward.mean()) - network.baseline)
