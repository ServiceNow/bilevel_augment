import torchvision.datasets as dset
import os
from pathlib import Path
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import numpy as np
from haven import haven_utils as hu


class CIFAR:
    def __init__(self, split, transform_lvl, datadir_base, name='cifar10', n_samples=None, colorjitter=False, val_transform='identity'):
        self.name = name
        if split in ['train', 'validation']:
            train = True
        elif split =='test':
            train = False
        self.split = split
        if self.name == 'cifar10':
            path = datadir_base or "/mnt/datasets/public/cifar10"
            self.n_classes = 10
            self.dataset = dset.CIFAR10(root=path, train=train, download=False)
        else:
            path = datadir_base or "/mnt/datasets/public/cifar100"
            self.n_classes = 100
            self.dataset = dset.CIFAR100(root=path, train=train, download=False)
        self.dataset.targets = np.array(self.dataset.targets)
        normalize = transforms.Normalize(mean = [x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std = [x / 255.0 for x in [63.0, 62.1, 66.7]])
        
        self.mean = normalize.mean
        self.std = normalize.std
        
        self.image_size = 32
        self.nc = 3
        
        if split == 'train':
            # Train transforms
            if transform_lvl == 0:
                transform = transforms.Compose([
                    # transforms.ToTensor(),
                    # normalize,
                ])

            elif transform_lvl == 1: 
                transform = transforms.Compose([
                    transforms.RandomCrop(self.image_size, padding=4),
                    # transforms.ToTensor(),
                    # normalize,
                ])

            elif transform_lvl == 1.5: 
                transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    # transforms.ToTensor(),
                    # normalize,
                ])

            elif transform_lvl == 2:
                transform = transforms.Compose([
                    transforms.RandomCrop(self.image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    # transforms.ToTensor(),
                    # normalize,
                ])

            elif transform_lvl == 2.5:
                transform = transforms.Compose([
                    transforms.RandomCrop(self.image_size, padding=4),
                    transforms.RandomAffine(10, translate=None, scale=(0.5, 2)),
                    # transforms.ToTensor(),
                    # normalize,
                ])

            elif transform_lvl == 3:
                transform = transforms.Compose([
                    transforms.RandomCrop(self.image_size, padding=4),
                    transforms.RandomAffine(10, translate=None, scale=(0.5, 2)),
                    transforms.RandomHorizontalFlip(),
                    # transforms.ToTensor(),
                    # normalize,
                ])
            else:
                raise ValueError('only lvls 0, 1, 1.5, 2, 2.5 and 3 are supported')

            if colorjitter:
                transform.transforms.append(
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                    )
                )
            transform.transforms.append(transforms.ToTensor())
            transform.transforms.append(normalize)

        elif split in ['validation', 'test']:
            # identity transform
            if val_transform == 'identity':
                transform = transforms.Compose([
                        transforms.ToTensor(),
                        normalize
                    ])
            elif val_transform == 'rotation':
                transform = transforms.Compose([
                        transforms.RandomRotation((45, 45)),
                        transforms.ToTensor(),
                        normalize
                    ])
            elif val_transform == 'translation':
                transform = transforms.Compose([
                        transforms.Pad((4, 4, 0, 0)),
                        transforms.CenterCrop(self.image_size),
                        transforms.ToTensor(),
                        normalize
                    ])
            elif val_transform == 'zoomin':
                transform = transforms.Compose([
                        transforms.Resize(int(self.image_size * 1.5)),
                        transforms.CenterCrop(self.image_size),
                        transforms.ToTensor(),
                        normalize
                    ])
            elif val_transform == 'zoomout':
                transform = transforms.Compose([
                        transforms.Resize(int(self.image_size * 0.75)),
                        transforms.Pad(4),
                        transforms.ToTensor(),
                        normalize
                    ])

            else:
                raise ValueError('%s is not supported' % val_transform)

        self.transform = transform

        # if n_samples is not None:
        #     with hu.random_seed(10):
        #         ind = np.random.choice(self.dataset.data.shape[0], n_samples, replace=False)
                
        #         self.dataset.data = self.dataset.data[ind]
        #         self.dataset.targets = np.array(self.dataset.targets)[ind]

        if n_samples is not None:
            assert n_samples % self.n_classes == 0, 'the number of samples %s must be a multiple of the number of classes %s' % (n_samples, self.n_classes)
            with hu.random_seed(10):
                n = int(n_samples/self.n_classes) # number of samples per class
                # Extract a balanced subset
                ind = np.hstack([np.random.choice(np.where(self.dataset.targets == l)[0], n, replace=False)
                      for l in np.unique(self.dataset.targets)])

                self.dataset.data = self.dataset.data[ind]
                self.dataset.targets = self.dataset.targets[ind]

    def get_labels(self):
        return self.dataset.targets
        # return [img[1] for img in self.dataset]

    def __getitem__(self, index):
        images_original, labels = self.dataset[index]
        images = self.transform(images_original)

        return {"images":images, 
                'labels':labels, 
                'meta':{'indices':index}}

    def __len__(self):
        return len(self.dataset)