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
        self.confidence_level = self.parameter("confidence_level", 0.5)
        self.interval = self.parameter("interval", 5)
        self.max_iter = self.parameter("max_iter", 100)
        
        self.reward = self.parameter("reward", 1)
        self.punish = self.parameter("punish", -1)

        self.low_confidence_interval = 0
        self.classes = self.parameter("classes", 2)

        network.decision = -1

    def forward(self, network) : 
        if(network.iteration < 2000) : 
            return

        ng_classes = network.find_objects("target")
        ng = ng_classes[0]

        tot = 0
        acts = []


        for i in range(self.classes) :
            a = int(80 * (i + 0))
            b = int(80 * (i + 1))

            act = 0
            if(type(ng.spikes) != type(True)) :
                act = torch.sum(ng.spikes[a:b], 0)
            acts.append(act)
            tot += act

        if(tot == 0) : 
            network.payoff = 0
            return 

        acts = torch.Tensor(acts)
        acts /= tot
        
        if(acts.max() < self.confidence_level and self.low_confidence_interval < self.max_iter) : 
            self.low_confidence_interval += 1
            network.payoff = 0
            return
        
        self.low_confidence_interval = 0
        network.decision = acts.argmax()

        if(network.decision == network.targets) : 
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
