import torch
import pymonntorch as pynt

from conex import *


class OnlineDataLoader(pynt.Behavior):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

