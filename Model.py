import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, bonetype):
        super(Model, self).__init__()

        
