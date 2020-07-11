from torch import nn
from .optimizers import get_optimizer
import torch 
from src import models
from torch.nn import functional as F
from src.utils import get_slope
import itertools 
import tqdm
from haven import haven_utils as hu
import os 
import numpy as np
from src import utils as ut
import torch.optim as optim


from .base_netC import lenet, lenet_meta
from .base_netC import resnet_meta, resnet_meta_old,resnet_cifar10_meta
from .base_netC import wide_resnet, wide_resnet_meta
import torchvision.models as models

from torchmeta.modules import MetaSequential, MetaLinear
from torchmeta.modules import DataParallel

class Classifier(nn.Module):
    def __init__(self, model_dict, dataset, device):
        super().__init__()
        
        self.dataset = dataset

        self.model_dict = model_dict
        if self.model_dict['name'] == 'resnet18':
            if self.model_dict['pretrained']:
                self.net = models.resnet18(pretrained=True)
                self.net.fc = nn.Linear(512, self.dataset.n_classes)
            else:
                self.net = models.resnet18(num_classes= self.dataset.n_classes)
        elif self.model_dict['name'] == 'resnet34':
            if self.model_dict['pretrained']:
                self.net = models.resnet34(pretrained=True)
                self.net.fc = nn.Linear(512, self.dataset.n_classes)
            else:
                self.net = models.resnet34(num_classes= self.dataset.n_classes)                
        elif self.model_dict['name'] == 'resnet50':
            if self.model_dict['pretrained']:
                self.net = models.resnet50(pretrained=True)
                self.net.fc = nn.Linear(512, self.dataset.n_classes)
            else:
                self.net = models.resnet50(num_classes= self.dataset.n_classes)

        elif self.model_dict['name'] == 'wide_resnet':
            self.net = wide_resnet.WideResNet(nc=3, depth=self.model_dict['RNDepth'], num_classes=self.dataset.n_classes, widen_factor=self.model_dict['RNWidth'], dropRate=self.model_dict['RNDO'])
        elif self.model_dict['name'] == 'lenet':
            self.net =  lenet.LeNet(nc=3, ncat= self.dataset.n_classes)
        elif self.model_dict['name'] == 'resnet20_cifar10_meta':
            self.net = resnet_cifar10_meta.resnet20()
        elif self.model_dict['name'] == 'resnet32_cifar10_meta':
            self.net = resnet_cifar10_meta.resnet32(num_classes= self.dataset.n_classes)
        elif self.model_dict['name'] == 'resnet56_cifar10_meta':
            self.net = resnet_cifar10_meta.resnet56(num_classes= self.dataset.n_classes)            
        elif self.model_dict['name'] == 'resnet18_meta':
            if self.model_dict.get('pretrained', True):
                self.net = resnet_meta.resnet18(pretrained=True)
                self.net.fc = MetaLinear(512, self.dataset.n_classes)
            else:
                self.net = resnet_meta.resnet18(num_classes= self.dataset.n_classes)
        elif self.model_dict['name'] == 'resnet18_meta_old':
                self.net = resnet_meta_old.ResNet18(nc=3, nclasses= self.dataset.n_classes)
        elif self.model_dict['name'] == 'resnet34_meta_old':
                self.net = resnet_meta_old.ResNet34(nc=3, nclasses= self.dataset.n_classes)
        elif self.model_dict['name'] == 'resnet34_meta':
            if self.model_dict.get('pretrained', True):
                self.net = resnet_meta.resnet34(pretrained=True)
                self.net.fc = MetaLinear(512, self.dataset.n_classes)
            else:
                self.net = resnet_meta.resnet34(num_classes= self.dataset.n_classes)   

        elif self.model_dict['name'] == 'resnet50_meta':
            if self.model_dict.get('pretrained', True):
                self.net = resnet_meta.resnet50(pretrained=True)
                self.net.fc = MetaLinear(512 * 4, self.dataset.n_classes)
            else:
                self.net = resnet_meta.resnet50(num_classes= self.dataset.n_classes)                  
        elif self.model_dict['name'] == 'wide_resnet_meta':
            self.net = wide_resnet_meta.WideResNet(nc=3, depth=self.model_dict['RNDepth'], num_classes=self.dataset.n_classes, widen_factor=self.model_dict['RNWidth'], dropRate=self.model_dict['RNDO'])

        elif self.model_dict['name'] == 'lenet_meta':
            self.net =  lenet_meta.LeNet(nc=3, ncat= self.dataset.n_classes)
        else:
            raise ValueError('network %s does not exist' % model_dict['name'])

        if (device.type == 'cuda'):
            self.net = DataParallel(self.net)
        self.net.to(device)
        # set optimizer
        self.opt_dict = model_dict['opt']
        self.lr_init = self.opt_dict['lr']
        if self.model_dict['opt']['name'] == 'sps':
            n_batches_per_epoch = 120
            self.opt = sps.Sps(self.net.parameters(), n_batches_per_epoch=n_batches_per_epoch, c=0.5, adapt_flag='smooth_iter', eps=0, eta_max=None)
        else:
            self.opt = optim.SGD(self.net.parameters(), 
                                lr=self.opt_dict['lr'], 
                                momentum=self.opt_dict['momentum'], 
                                weight_decay=self.opt_dict['weight_decay'])

        # variables
        self.device = device

    def get_state_dict(self):
        state_dict = {'net': self.net.state_dict(),
                      'opt': self.opt.state_dict(),
                      }

        return state_dict

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict['net'])
        self.opt.load_state_dict(state_dict['opt'])

    def on_trainloader_start(self, epoch):
        if self.opt_dict['sched']:
            ut.adjust_learning_rate_netC(self.opt, epoch, self.lr_init, self.model_dict['name'], self.dataset.name)

    def train_on_batch(self, batch):
        images, labels = batch['images'].to(self.device, non_blocking=True), batch['labels'].to(self.device, non_blocking=True)   
     
        logits = self.net(images)
        loss = F.cross_entropy(logits, labels, reduction="mean")

        self.opt.zero_grad()
        loss.backward()  

        if self.opt_dict['name'] == 'sps':
            self.opt.step(loss=loss)
        else:
            self.opt.step()
        # print(ut.compute_parameter_sum(self))

        return loss.item()