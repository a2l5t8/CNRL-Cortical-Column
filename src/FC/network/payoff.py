from pymonntorch import *
from conex import *
import torch


class ConfidenceLevelPayOff(Payoff) : 

    """
    General Description of ConfidenceLevelPayOff Behavior.
    """

    def __init__(self, *args, initial_payoff=0.0, **kwargs):
        super().__init__(*args, initial_payoff=initial_payoff, **kwargs)

    def initialize(self, network) : 

        """
        Args : 
            confidence_level (float) : the percentage of the the maximum population activity to make a decision, if none has reached the threshold, is does not change payoff.
            interval (int) : to be added
            max_iter (int) : to be added
            reward (float) : to be added
            punish (float) : to be added
        """
        
        super().initialize(network)
        self.confidence_level = self.parameter("confidence_level", 0.6)
        self.interval = self.parameter("interval", 5)
        self.max_iter = self.parameter("max_iter", 200)
        
        self.reward = self.parameter("reward", 1)
        self.punish = self.parameter("punish", -1)

    def forward(self, network) : 
        ng_classes = network.find_objects("target")

        tot = 0
        acts = []
        for ng in ng_classes :
            act = torch.sum(ng["spikes", 0][:,0] > max(0, network.iteration - self.interval), 0)
            acts.append(act)
            tot += act

        acts = torch.Tensor(acts)
        acts /= tot

        if(max(acts) < self.confidence_level) : 
            network.payoff = 0
            return
        
        prediction = acts.argmax()
        if(prediction == network.target) : 
            network.payoff = self.reward
        else :
            network.payoff = self.punish
        

class TimeWindowPayOff(Payoff) : 

    def __init__(self, *args, initial_payoff=0.0, **kwargs):
        super().__init__(*args, initial_payoff=initial_payoff, **kwargs)

    def initialize(self, network) : 
        super().initialize(network)

        self.time_window = self.parameter("time_window", 100)

    def forward(self, network) : 
        pass
