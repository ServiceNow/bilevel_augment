import torch
from collections import OrderedDict
import math


def get_slope(slope_annealing, epoch):
    if slope_annealing:
        slope = 1.0 * (1.005 ** (epoch -1))
    else:
        slope = 1.0
    
    return slope

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)
#     elif classname.find('Linear') != -1:
#         m.weight.data.normal_(0.0, 0.05)
#         m.bias.data.fill_(0)

def adjust_learning_rate_netC(optimizer, epoch, lr_init, model, dataset, return_lr=False):

    if model in ['resnet18_meta', 'resnet18_meta_2']:
        if dataset in ['bach']:
            # eta_min = 0.0001
            # T_max = 20
            # lr = eta_min + (lr_init - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2
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
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = lr_init * (0.1 ** (epoch // 30))
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