import torchvision.datasets as dset
import os
from pathlib import Path
from torch.utils.data import Dataset
import torch
from torch.utils.data import sampler

import torchvision.transforms as transforms
import numpy as np
from . import cifar
from haven import haven_utils as hu
from src.datasets.utils.patch_extractors import *

from sklearn.model_selection import train_test_split

def get_dataset(datasetparams, split, exp_dict, datadir_base=None, 
                n_samples=None, transform_lvl=None, colorjitter=False, val_transform='identity'):
    if datasetparams['name'] == 'cifar10':
        return cifar.CIFAR(split=split, transform_lvl=transform_lvl, 
                        datadir_base=datadir_base, name=datasetparams['name'],
                        n_samples=n_samples,
                        colorjitter=colorjitter,
                        val_transform=val_transform)
    else:
        raise ValueError("dataset %s does not exist" % datasetparams['name'])


# ===============================
# loaders
def get_train_val_dataloader(exp_dict, 
                             trainset,
                             valset,
                             mixtrainval=True, 
                             pin_memory=False, 
                             num_workers=0):
    indices = np.arange(len(trainset))
    # labels=[img[1] for img in trainset.imgs]
    labels = trainset.get_labels()

    # n_total = len(trainset)
    # n_train = int((1 - exp_dict['valratio']) * n_total)
    # n_val = n_total - n_train

    # if mixtrainval:
    #     ind_train, ind_val = get_train_val_ind(indices, labels, exp_dict['valratio'], mixtrainval, exp_dict['fixedSeed'])
    # else:
    #     with hu.random_seed(exp_dict['fixedSeed']):
    #         ind_train, ind_val = get_train_val_ind(indices, n_train, n_val)

    ind_train, ind_val = get_train_val_ind(indices, labels, exp_dict['valratio'], mixtrainval, exp_dict['fixedSeed'])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=exp_dict['batch']['size'], 
                                            sampler=SaypraSampler(ind=ind_train),
                                            shuffle=False, 
                                            num_workers=num_workers, 
                                            pin_memory=pin_memory,
                                            drop_last=False)

    valloader = torch.utils.data.DataLoader(valset, batch_size=exp_dict['batch']['size'],
                                        sampler=SaypraSampler(ind=ind_val),
                                        num_workers=num_workers, 
                                        pin_memory=pin_memory,
                                        drop_last=False)
    return trainloader, valloader


def get_train_val_ind(indices, labels, valratio, shuffle, seed):
    # ind_train = np.random.choice(indices, n_train, replace=False) 
    # ind_remaining = np.setdiff1d(indices, ind_train)
    # ind_val = np.random.choice(ind_remaining, n_val, replace=False) 

    ind_train, ind_val, _, _ = train_test_split(indices, labels, test_size=valratio, random_state= seed, shuffle=shuffle)

    return ind_train, ind_val
class SaypraSampler(sampler.Sampler):
    def __init__(self, ind):
        self.ind = ind

    def __iter__(self):
        return iter(torch.LongTensor(self.ind))

    def __len__(self):
        return len(self.ind)



