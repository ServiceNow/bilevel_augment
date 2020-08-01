from torch import nn
from .optimizers import get_optimizer
import torch 
from src import models
from src.utils import get_slope
import itertools 
import tqdm
from haven import haven_utils as hu
import os 
import numpy as np
from src import utils as ut
import torch.optim as optim
from . import netA
from . import netC

import gc

class Blvl(nn.Module):
    def __init__(self, model_dict, dataset, device):
        super().__init__()

        self.clf_loss = False
        self.netC = netC.Classifier(model_dict['netC'], dataset=dataset,  device=device)
        if model_dict.get('netA') is not None:
            self.netA = netA.Augmenter(model_dict['netA'], dataset=dataset, device=device)
            if model_dict['netA'].get('clf_loss'):
                self.clf_loss = True

        # variables 
        self.device = device 
        self.model_dict = model_dict

    def get_state_dict(self):
        state_dict = {'netC':self.netC.get_state_dict()}
        if self.model_dict.get('netA') is not None:
            state_dict['netA'] = self.netA.get_state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        self.netC.load_state_dict(state_dict['netC'])
        if self.model_dict.get('netA') is not None:
            self.netA.load_state_dict(state_dict['netA'])

    def vis_on_loader(self, split, dataset, savedir_images, epoch):
        self.eval()
        batch = torch.utils.data.dataloader.default_collate([dataset[i] for i in range(5)])
        self.vis_on_batch(split, batch, savedir_images, epoch)

    def train_on_loader(self, loader, valloader, epoch, exp_dict):
        self.train()

        # Update learning rate
        self.netC.on_trainloader_start(epoch)
        if self.model_dict.get('netA') is not None:
            self.netA.on_trainloader_start(epoch, valloader, self.netC)
        
        loss_sum = 0.
        loss_count = 0
        transformations_mean = 0.
        transformations_std = 0.
        transforms = None
        with tqdm.tqdm(total=len(loader), leave=False) as pbar:
            for batch in loader:
                
                if self.model_dict.get('netA') is not None:
                    # if augmenter, train netC with netA
                    loss, transformations = self.netA.train_on_batch(batch, netC=self.netC)
                    if transforms is None:
                        transforms = transformations.detach()
                    else:
                        transforms = torch.cat((transforms, transformations.detach()))

                    if self.clf_loss:
                        loss += float(self.netC.train_on_batch(batch))
                    del transformations
                    
                else:
                    # if no augmenter, just train netC
                    loss = float(self.netC.train_on_batch(batch))

                loss_sum += loss
                del loss
                del batch
                gc.collect()
                loss_count += 1
                train_loss = loss_sum / loss_count

                pbar.set_description("Training - Loss: %.3f" % (train_loss))
                pbar.update(1)

            if self.model_dict.get('netA') is not None:
                transformations_mean = torch.mean(transforms, dim=0)
                transformations_std = torch.std(transforms, dim=0)
                del transforms
                gc.collect()

        
        if self.model_dict.get('netA') is not None:
            del self.netA.moms
            gc.collect()
            if self.netC.opt_dict['name'] == 'sps':
                return {'loss': train_loss, 'netC_lr': self.netC.opt.state['step_size'], 'transformations_mean': transformations_mean, 'transformations_std': transformations_std}
            else:
                return {'loss': train_loss, 'netC_lr': ut.adjust_learning_rate_netC(self.netC.opt, epoch, self.netC.opt.defaults['lr'], self.netC.model_dict['name'], self.netC.dataset.name, return_lr=True), 'transformations_mean': transformations_mean, 'transformations_std': transformations_std}
        else:
            if self.netC.opt_dict['name'] == 'sps':
                return {'loss': train_loss, 'netC_lr': self.netC.opt.state['step_size']}
            else:
                return {'loss': train_loss, 'netC_lr': ut.adjust_learning_rate_netC(self.netC.opt, epoch, self.netC.opt.defaults['lr'], self.netC.model_dict['name'], self.netC.dataset.name, return_lr=True)}
    @torch.no_grad()
    def vis_on_batch(self, split, batch, savedir_images, epoch):
        self.eval()
        images = batch['images'].to(self.device)
        labels = batch['labels'].to(self.device)
        mean = torch.tensor(self.netC.dataset.mean).to(self.device)
        std = torch.tensor(self.netC.dataset.std).to(self.device)
        org = images.detach()
        org = (org * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)).cpu().numpy()
        
        img_list = []
        if split == 'train' and self.model_dict.get('netA') is not None:
            augimages, _ , _ = self.netA.apply_augmentation(images, labels)
            aug = augimages.detach()
            aug = (aug * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)).cpu().numpy()
            for i in range(org.shape[0]):
                both = np.concatenate([org[i], aug[i]], axis=1)
                img_list += [both]
            img_list = np.concatenate(img_list, axis=2)
        else:
            img_list = np.concatenate(org, axis=2)
        hu.save_image(os.path.join(savedir_images, '%s_%s.jpg' % (split, epoch)),  img_list)

    @torch.no_grad()
    def test_on_loader(self, loader):
        self.eval()

        clf_monitor = ClfMonitor()
        with tqdm.tqdm(total=len(loader), leave=False) as pbar:
            for batch in loader:
                images, labels = batch['images'].to(self.device), batch['labels'].to(self.device)
                # Calculate scores
                logits = self.netC.net(images)
                # compute metrics (loss and classification)
                clf_monitor.add(logits, labels)
                pbar.set_description("Validating - Acc: %.3f" % (clf_monitor.get_avg_scores()['acc']))
                pbar.update(1)
                del logits
                del images
                del labels
                del batch
                gc.collect()

        score_dict = clf_monitor.get_avg_scores()

        return score_dict

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        # res.append(correct_k.mul_(100.0 / batch_size))
        res.append(correct_k)
    return res


class ClfMonitor(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.n_corr = 0
        self.n_corr5 = 0
        self.loss_sum = 0.0
        self.n_train = 0.

    @torch.no_grad()
    def add(self, logits, labels):
        _, preds = torch.max(logits, 1)
        self.n_corr += (preds == labels).sum().item()
        self.n_train += labels.size(0)

            
    def get_avg_scores(self):
        acc = self.n_corr / self.n_train
        return {'acc':100. * acc} 