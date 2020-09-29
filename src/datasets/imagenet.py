import torchvision.datasets as dset
import os
from pathlib import Path
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import numpy as np
from haven import haven_utils as hu
from PIL import Image 

class ImageNet:
    def __init__(self, split, transform_lvl, datadir_base, n_samples=None, colorjitter=False, val_transform='identity', netA=None):
        path = datadir_base or '/mnt/datasets/public/imagenet/imagenet-data/raw-data/'
        self.name = 'imagenet'
        self.n_classes = 1000
        self.image_size = 224
        self.nc = 3
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.mean = normalize.mean
        self.std = normalize.std

        if split == 'train':
            if transform_lvl == 0:
                transform = transforms.Compose([
                    # transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    # transforms.ToTensor(),
                    # normalize,
                ])
                if netA is not None:
                    transform.transforms.append(netA)

            elif transform_lvl == 1: 
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),                    
                    transforms.RandomCrop(self.image_size, padding=4),
                    # transforms.ToTensor(),
                    # normalize,
                ])

            elif transform_lvl == 1.5: 
                transform = transforms.Compose([
                    # transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(),
                    # transforms.ToTensor(),
                    # normalize,
                ])

            elif transform_lvl == 2:
                transform = transforms.Compose([
                    # transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),                    
                    transforms.RandomCrop(self.image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    # transforms.ToTensor(),
                    # normalize,
                ])
            
            elif transform_lvl == 2.5:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),                    
                    transforms.RandomCrop(self.image_size, padding=4),
                    transforms.RandomAffine(10, translate=None, scale=(0.5, 2)),
                    # transforms.ToTensor(),
                    # normalize,
                ])

            elif transform_lvl == 3:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
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
                        brightness=0.5,
                        contrast=0.5,
                        saturation=0.5,
                    )
                )
            transform.transforms.append(transforms.ToTensor())
            transform.transforms.append(normalize)

        elif split in ['validation', 'test']:
            # identity transform
            transform = transforms.Compose([
                    # transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])

        self.transform = transform
        
        if split in ['train', 'validation']:
            fname = '/mnt/projects/bilvlda/dataset/imagenet/imagenet_train.json'

            if not os.path.exists(fname):
                dataset = dset.ImageFolder(root=os.path.join(path, 'train'))
                hu.save_json(fname, dataset.imgs)

            self.imgs = np.array(hu.load_json(fname))
            assert(len(self.imgs) == 1281167)

        elif split =='test':
            fname = '/mnt/projects/bilvlda/dataset/imagenet/imagenet_validation.json'

            if not os.path.exists(fname):
                dataset = dset.ImageFolder(root=os.path.join(path, 'validation'))
                hu.save_json(fname, dataset.imgs)
            self.imgs = np.array(hu.load_json(fname))    
            assert(len(self.imgs) == 50000)

        if n_samples is not None:
            assert n_samples % self.n_classes == 0, 'the number of samples %s must be a multiple of the number of classes %s' % (n_samples, self.n_classes)
            with hu.random_seed(10):
                imgs = np.array(self.imgs)
                n = int(n_samples/self.n_classes) # number of samples per class
                # Extract a balanced subset
                ind = np.hstack([np.random.choice(np.where(imgs[:,1] == l)[0], n, replace=False)
                      for l in np.unique(imgs[:,1])])
                # ind = np.random.choice(imgs.shape[0], n_samples, replace=False)
                
                self.imgs = imgs[ind]

    def get_labels(self):
        return np.array([img[1] for img in self.imgs])

    def __getitem__(self, index):
        image_path, labels = self.imgs[index]
        images_original = Image.open(image_path)
        images_original = images_original.convert('RGB')
        images = self.transform(images_original)

        return {"images":images, 
                'labels':int(labels), 
                'meta':{'indices':index}}

    def __len__(self):
        return len(self.imgs)