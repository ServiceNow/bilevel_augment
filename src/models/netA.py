from torch import nn
from .optimizers import get_optimizer
import torch 

import itertools 
import tqdm
from haven import haven_utils as hu
import os 
import numpy as np

import torch.optim as optim
from .base_netA import stn, small_affine 

from torch.nn import functional as F
from collections import OrderedDict
from .base_netA.det_utils import get_slope
from  src import utils as ut

from torchmeta.modules import DataParallel

import gc

class Augmenter(nn.Module):
    def __init__(self, model_dict, dataset, device):
        super().__init__()

        if model_dict['name'] == 'stn':
            self.net = stn.STN(isize=dataset.image_size,
                                    n_channels=dataset.nc, 
                                    n_filters=64, 
                                    nz=100, 
                                    datasetmean=dataset.mean, 
                                    datasetstd=dataset.std)

        elif model_dict['name'] == 'small_affine':
            self.net = small_affine.smallAffine(nz=6, 
                                            transformation=model_dict['transform'], 
                                            datasetmean=dataset.mean, 
                                            datasetstd=dataset.std)

        else:
            raise ValueError('network %s does not exist' % model_dict['name'])

        if (device.type == 'cuda'):
            self.net = DataParallel(self.net)

        self.net.to(device)

        self.device = device
        self.factor = model_dict['factor']
        self.name = model_dict['name']
         
        if model_dict['name'] != 'random_augmenter':
            self.opt_dict = model_dict['opt']   
            self.lr_init = self.opt_dict['lr']            
            self.opt = optim.SGD(self.net.parameters(), 
                                 lr=self.opt_dict['lr'], 
                                 momentum=self.opt_dict['momentum'], 
                                 weight_decay=self.opt_dict['weight_decay'])

    def cycle(self, iterable):
        iterator = iter(iterable)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(iterable)

    def get_state_dict(self):
        state_dict = {}
        if hasattr(self, 'opt'):
            state_dict['net'] =  self.net.state_dict()
            state_dict['opt'] = self.opt.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        if hasattr(self, 'opt'):
            self.net.load_state_dict(state_dict['net'])
            self.opt.load_state_dict(state_dict['opt'])
    
    def apply_augmentation(self, images, labels):
        # apply augmentation to the given images
        factor = self.factor
        if factor > 1:
            labels = labels.repeat(factor)  
            images = images.repeat(factor, 1, 1, 1)

        with torch.autograd.set_detect_anomaly(True):
            if self.name in ['mask_cnn', 'mask_fc', 'color_mask']:
                augimages, _, transformations = self.net(images)

            elif self.name in ['affine_color_mask_det']:
                augimages, transformations = self.net(images, get_slope(self.slope_annealing, self.epoch))

            else:
                augimages, transformations = self.net(images)

        return augimages, labels, transformations

    def on_trainloader_start(self, epoch, valloader, netC):
        # Get slope
        if hasattr(self, 'slope_annealing'):
            self.slope = get_slope(self.slope_annealing, epoch)
        # Update optimizer
        if self.opt_dict['sched']:
            ut.adjust_learning_rate_netA(self.optimizerA, epoch, self.lrA_init)

        # initialize momentums
        if netC.opt.defaults['momentum']:
            self.moms = OrderedDict()
            for (name, p) in netC.net.named_parameters():
                self.moms[name] = torch.zeros(p.shape).to(self.device)

        self.epoch = epoch
        # Cycle through val_loader
        self.val_gen = self.cycle(valloader)

    def train_on_batch(self, batch, netC):
        self.train()
        images, labels = batch['images'].to(self.device, non_blocking=True), batch['labels'].to(self.device, non_blocking=True)
        images, labels, transformations = self.apply_augmentation(images, labels)

        # Use classifier 
        logits = netC.net(images)
        loss_clf = F.cross_entropy(logits, labels, reduction="mean")
        
        netC.opt.zero_grad()

        if self.name in ['random_augmenter']:
            # Update the classifier only 
            loss_clf.backward() 
            netC.opt.step()

            return loss_clf

        elif self.name in ['stn']:
            # Update the style transformer network 
            self.opt.zero_grad()
            loss_clf.backward() 
            netC.opt.step()
            self.opt.step()

            return loss_clf
        
        else:
            # Update the augmenter through a validation batch
            # Calculate new weights w^t+1 to calculate the validation loss
            batch_val = next(self.val_gen)
            valimages, vallabels = batch_val['images'].to(self.device, non_blocking=True), batch_val['labels'].to(self.device, non_blocking=True)

            # construct graph
            loss_clf.backward(create_graph=True, retain_graph=True)  
         
            # for p in netC.net.parameters():
            #     p.requires_grad = False  # freeze C

            self.w_t_1 = OrderedDict()
            lr = ut.adjust_learning_rate_netC(netC.opt, self.epoch, netC.lr_init, netC.model_dict['name'], netC.dataset.name, return_lr=True) # get step size
 
            if netC.opt.defaults['momentum']:
                # for name in self.moms:
                #     self.moms[name].detach_()
                
                for (name, p) in netC.net.named_parameters():
                    p.requires_grad = False
                    self.moms[name].detach_()                    
                    self.moms[name] = netC.opt.defaults['momentum'] * self.moms[name] + p.grad # update momentums
                    self.w_t_1[name] = p - lr * self.moms[name] # compute future weights
                
            else:
                for (name, p) in netC.net.named_parameters():
                    p.requires_grad = False
                    self.w_t_1[name] = p - lr * p.grad # compute future weights

            # Calculate validation loss
            valoutput = netC.net(valimages, params=self.w_t_1)
            loss_aug = F.cross_entropy(valoutput, vallabels, reduction='mean')
            self.opt.zero_grad()
            loss_aug.backward()  
            self.opt.step()
            del self.w_t_1

            
            # After gradient is computed for A, unfreeze C
            for p in netC.net.parameters():
                p.requires_grad = True
            netC.opt.step()

            del images
            del labels
            gc.collect()

            return float(loss_clf.item()), transformations


    def cycle(self, iterable):
        iterator = iter(iterable)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(iterable)

    def __call__(self, img):
        img = img.unsqueeze(0)
        img, _ = self.net.forward(img)
        img = img.squeeze(0)
        return img


