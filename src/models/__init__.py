import torchvision.models as models
import tqdm 
import torch
from torch import nn

from . import blvl

def get_model(exp_dict, dataset, device):
   # classifier models
    if exp_dict['model']['name'] == 'blvl':
        model = blvl.Blvl(exp_dict['model'], dataset, device)
        return model


