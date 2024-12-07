import torch
from conex import *
from pymonntorch import *

class AttentionBasedRSTDP(SimpleRSTDP) : 

    """
    Attention-Based Reward-modulated Spike-Timing Dependent Plasticity (RSTDP) rule for simple connections.

    Note: The implementation uses local variables (spike trace).

    Args:
        a_plus (float): Coefficient for the positive weight change. The default is None.
        a_minus (float): Coefficient for the negative weight change. The default is None.
        tau_c (float): Time constant for the eligibility trace. The default is None.
        init_c_mode (int): Initialization mode for the eligibility trace. The default is 0.

        attention_plus (float) : Coefficient for the attention of the decision pop. The default is 1.
        attention_minus (float) : Coefficient for the attention of other pops. The default is 0.
    """

    def __init__(
        self,
        a_plus,
        a_minus,
        tau_c,
        *args,
        init_c_mode=0,
        w_min=0.0,
        w_max=1.0,
        attention_plus = 1,
        attention_minus = 0,
        positive_bound=None,
        negative_bound=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            a_plus=a_plus,
            a_minus=a_minus,
            tau_c=tau_c,
            init_c_mode=init_c_mode,
            w_min=w_min,
            w_max=w_max,
            positive_bound=positive_bound,
            negative_bound=negative_bound,
            **kwargs,
        )

    def initialize(self, synapse) :
        super().initialize(synapse)

        self.attention_plus = self.parameter("attention_plus", 1)
        self.attention_minus = self.parameter("attention_minus", 0)

        self.k = self.parameter("number_of_classes", required=True)


    def forward(self, synapse) :
        computed_stdp = self.compute_dw(synapse)

        """ Attention """
        sz = synapse.dst.size
        att = torch.ones(sz) * self.attention_minus

        dec = synapse.network.decision
        if(dec == -1) :
            att = torch.ones(sz) * self.attention_plus
        else :
            st = int(dec * sz/self.k)
            en = int((dec + 1) * sz/self.k)

            att[st:en] = self.attention_plus

        attention_mat = att.expand((synapse.src.size, -1))
        """ --------- """

        synapse.c += (-synapse.c / self.tau_c) + computed_stdp * attention_mat
        synapse.weights += synapse.c * synapse.network.dopamine_concentration