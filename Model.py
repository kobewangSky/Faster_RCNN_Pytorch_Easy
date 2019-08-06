import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from backbone.BackBoneInterface import backboneinterface

class Model(nn.Module):
    def __init__(self, backbone:backboneinterface, classnum):
        super(Model, self).__init__()
        self.backbone = backbone.feature()



    def forward(self, ):


        
