import torch
from collections import OrderedDict
import math

def adjust_learning_rate_netC(optimizer, epoch, lr_init, model, dataset, return_lr=False):

    if model in ['resnet18_meta', 'resnet18_meta_2']:
        if dataset in ['bach']:
            if epoch < 6:
                lr = lr_init
            elif epoch < 10:
                lr = lr_init * 0.1
            elif epoch < 20:
                lr = lr_init * 0.1 * 0.1
            else:
                lr = lr_init * 0.1 * 0.1 * 0.1
        else:
            if epoch < 60:
                lr = lr_init
            elif epoch < 120:
                lr = lr_init * 0.2
            elif epoch < 160:
                lr = lr_init * 0.2 * 0.2
            else:
                lr = lr_init * 0.2 * 0.2 * 0.2
    if return_lr:
            return lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def adjust_learning_rate_netA(optimizer, epoch, lr_init, return_lr=False):
    if epoch < 100:
        lr = lr_init
    elif epoch < 150:
        lr = lr_init / 2
    else:
        lr = lr_init / 10
    if return_lr:
        return lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr