from pymonntorch import *
from conex import SimpleSTDP

class FasterSTDP(SimpleSTDP):
    def forward(self, sg):
        sg.weights[sg.pre_spike, :] -= (
            sg.post_trace[None, :]
            * self.a_minus
            * self.n_bound(sg.weights[sg.pre_spike, :], self.w_min, self.w_max)
        )
        sg.weights[:, sg.post_spike] += (
            sg.pre_trace[:, None]
            * self.a_plus
            * self.p_bound(sg.weights[:, sg.post_spike], self.w_min, self.w_max)
        )