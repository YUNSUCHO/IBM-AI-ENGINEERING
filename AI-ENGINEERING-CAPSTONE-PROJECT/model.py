# These are the libraries will be used for this lab.
import torchvision.models as models
from PIL import Image
import pandas
from torchvision import transforms
import torch.nn as nn
import time
import torch 
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import glob
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
from PIL import Image
import pandas as pd
import os
torch.manual_seed(0)


def mymodel(modelname = 'resnet18'):
    if modelname == 'resnet18':
        model = models.resnet18(pretrained = True)
    elif modelname == 'vgg16':
        model = models.vgg16(pretrained = True)
    else :
        print("model doest not exist")
        model = models.resnet18(pretrained = True)
    return model


