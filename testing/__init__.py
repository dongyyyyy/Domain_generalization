from models.ResNet import *
from utils.function import *
from utils.dataloader_custom import *

from info.OfficeHome_info import*
from info.PACS_info import *
from utils.loss_fn import *
from tqdm import tqdm


import torch
import torchvision.models as models
from models.ResNet import *
from torchsummary import summary
from torch.utils.data import DataLoader


import time
import sys
import os
import random
import numpy as np
import multiprocessing
from models.modules.MixStyle import *

def deactivate_mixstyle(m):
    if type(m) == MixStyle:
        m.set_activation_status(False)

def activate_mixstyle(m):
    if type(m) == MixStyle:
        m.set_activation_status(True)

