import numpy as np
import torchvision.utils
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import einops
import torch
import torch.nn as nn
from torch.optim import Adam

import torch
from PIL import Image
from scipy.io import loadmat


def my_weighted_mse_loss1(x, y, cond):
    mse = nn.MSELoss()
    myweight = cond - torch.min(cond)
    myweight = myweight / torch.max(myweight)
    loss = mse(x * myweight, y * myweight)
    return loss
