import torchvision.datasets as dset
import os
from pathlib import Path
from torch.utils.data import Dataset
import torch
from torch.utils.data import sampler

import torchvision.transforms as transforms
import numpy as np
from . import cifar, imagenet, tinyimagenet, bach
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
    elif datasetparams['name'] == 'cifar100':
        return cifar.CIFAR(split=split, transform_lvl=transform_lvl, 
                        datadir_base=datadir_base, name=datasetparams['name'],
                        n_samples=n_samples,
                        colorjitter=colorjitter,
                        val_transform=val_transform)
    elif datasetparams['name'] == 'imagenet':
        return imagenet.ImageNet(split=split, transform_lvl=transform_lvl, 
                                datadir_base=datadir_base,
                                n_samples=n_samples,
                                val_transform=val_transform)
        return dataset
    elif datasetparams['name'] == 'tinyimagenet':
        return tinyimagenet.TinyImageNet(split=split, transform_lvl=transform_lvl, 
                                datadir_base=datadir_base,
                                n_samples=n_samples,
                                val_transform=val_transform)                                                        
    elif datasetparams['name'] == 'bach':
        return bach.Bach(split=split, transform_lvl=transform_lvl, 
                                datadir_base=datadir_base,
                                folds_path=datasetparams['folds_path'],
                                fold=datasetparams['fold'],
                                patch_size=512, 
                                patch_extractor=NoOverlap,
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

    labels = trainset.get_labels()

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

    ind_train, ind_val, _, _ = train_test_split(indices, labels, test_size=valratio, random_state= seed, shuffle=shuffle)

    return ind_train, ind_val
class SaypraSampler(sampler.Sampler):
    def __init__(self, ind):
        self.ind = ind

    def __iter__(self):
        return iter(torch.LongTensor(self.ind))

    def __len__(self):
        return len(self.ind)



