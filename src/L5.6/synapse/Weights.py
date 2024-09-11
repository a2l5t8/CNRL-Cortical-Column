import numpy as np
from matplotlib import pyplot as plt
import random
import math
import pandas as pd
import tqdm
import seaborn as sns

from pymonntorch import *
from conex import *

class WeightInitializerAncher(Behavior) : 

    """
    WeightInitializer to to only ancher neurons at the center of GPCell reference frame.

    Args :
        (float) R : Radius of the neurons to be activated. Default is 1.
        (float) w : The synaptic weight of connections, default is 20.
    """

    def initialize(self, synapse) : 

        self.R = self.parameter("R", 1)
        self.w = self.parameter("w", 20)

        synapse.weights = synapse.matrix("zeros")
        synapse.weights[0][(synapse.dst.x**2 + synapse.dst.y**2) <= self.R**2] = self.w