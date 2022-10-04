import numpy as np
import torch.nn as nn 
import torch.optim as optim 
import torchvision.transforms as transforms
# from info.OfficeHome_info import *
import os 
from PIL import Image
import random
import torch.nn.functional as F

class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1)

        return b.mean()