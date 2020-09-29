import torchvision.datasets as dset
import os
from pathlib import Path
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import numpy as np
from haven import haven_utils as hu
from PIL import Image 

# from .utils.tinyimagenet_dataset import TinyImageNet

class TinyImageNet:
    def __init__(self, split, transform_lvl, datadir_base, n_samples=None, val_transform='identity'):
        path = datadir_base or '/mnt/projects/bilvlda/dataset/tiny-imagenet-200'
        self.name = 'tinyimagenet'
        self.n_classes = 200
        self.image_size = 64
        self.nc = 3
        
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])

        self.mean = normalize.mean
        self.std = normalize.std

        if split == 'train':
            if transform_lvl == 0:
                transform = transforms.Compose([
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    normalize,
                ])
            
            elif transform_lvl == 1: 
                transform = transforms.Compose([
                    transforms.RandomCrop(self.image_size, padding=4),
                    transforms.ToTensor(),
                    normalize,
                ])

            elif transform_lvl == 1.5: 
                transform = transforms.Compose([
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])

            elif transform_lvl == 2:
                transform = transforms.Compose([
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.RandomCrop(self.image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])

            elif transform_lvl == 2.5:
                transform = transforms.Compose([
                    transforms.RandomCrop(self.image_size, padding=4),
                    transforms.RandomAffine(10, translate=None, scale=(0.5, 2)),
                    transforms.ToTensor(),
                    normalize,
                ])

            elif transform_lvl == 3:
                transform = transforms.Compose([
                    transforms.RandomCrop(self.image_size, padding=4),
                    transforms.RandomAffine(10, translate=None, scale=(0.5, 2)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])

            else:
                raise ValueError('only lvls 0, 1, 1.5, 2, 2.5 and 3 are supported')
        
        elif split in ['validation', 'test']:
            # identity transform
            if val_transform == 'identity':
                transform = transforms.Compose([
                        transforms.Lambda(lambda x: x.convert("RGB")),
                        transforms.ToTensor(),
                        normalize
                    ])
            elif val_transform == 'rotation':
                transform = transforms.Compose([
                        transforms.Lambda(lambda x: x.convert("RGB")),
                        transforms.RandomRotation((45, 45)),
                        transforms.ToTensor(),
                        normalize
                    ])
            elif val_transform == 'translation':
                transform = transforms.Compose([
                        transforms.Lambda(lambda x: x.convert("RGB")),                    
                        transforms.Pad((4, 4, 0, 0)),
                        transforms.CenterCrop(self.image_size),
                        transforms.ToTensor(),
                        normalize
                    ])
            elif val_transform == 'zoomin':
                transform = transforms.Compose([
                        transforms.Lambda(lambda x: x.convert("RGB")),
                        transforms.Resize(int(self.image_size * 1.5)),
                        transforms.CenterCrop(self.image_size),
                        transforms.ToTensor(),
                        normalize
                    ])
            elif val_transform == 'zoomout':
                transform = transforms.Compose([
                        transforms.Lambda(lambda x: x.convert("RGB")),
                        transforms.Resize(int(self.image_size * 0.75)),
                        transforms.Pad(4),
                        transforms.ToTensor(),
                        normalize
                    ])

        self.transform = transform
        
        if split in ['train', 'validation']:
            fname = '/mnt/projects/bilvlda/dataset/tiny-imagenet-200/tinyimagenet_train.json'

            if not os.path.exists(fname):
                dataset = dset.ImageFolder(root=os.path.join(path, 'train'))
                hu.save_json(fname, dataset.imgs)

            self.imgs = np.array(hu.load_json(fname))
            assert(len(self.imgs) == 100000)

        elif split =='test':
            fname = '/mnt/projects/bilvlda/dataset/tiny-imagenet-200/tinyimagenet_validation.json'

            if not os.path.exists(fname):
                dataset = dset.ImageFolder(root=os.path.join(path, 'val'))
                hu.save_json(fname, dataset.imgs)
            self.imgs = np.array(hu.load_json(fname)) 
            assert(len(self.imgs) == 10000)

        if n_samples is not None:
            with hu.random_seed(10):
                imgs = np.array(self.imgs)
                ind = np.random.choice(imgs.shape[0], n_samples, replace=False)
                self.imgs = imgs[ind]

    def get_labels(self):
        return np.array([img[1] for img in self.imgs])

    def __getitem__(self, index):
        image_path, labels = self.imgs[index]
        images_original = Image.open(image_path)
        images = self.transform(images_original)

        return {"images":images, 
                'labels':int(labels), 
                'meta':{'indices':index}}

    def __len__(self):
        return len(self.imgs)