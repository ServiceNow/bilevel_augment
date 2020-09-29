import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform
from .color_utils import *
# import kornia

class affineColor(nn.Module):
    def __init__(self, nz, datasetmean, datasetstd, neurons=10):
        super(affineColor, self).__init__()
        self.nz = nz
        self.mean = torch.tensor(datasetmean)
        self.std = torch.tensor(datasetstd)
        self.lin1 = nn.Linear(self.nz, neurons)
        self.lin2 = nn.Linear(neurons, 10 * neurons)
        self.lin3 = nn.Linear(10*neurons, 10)
        self.drop = nn.Dropout(0.2)
        # self.buffer_in = torch.Tensor()
        # self.buffer_out = torch.Tensor()

    def get_transformation_parameters(self, noise):
        transparams = F.relu(self.lin1(noise))
        transparams = self.drop(transparams)
        transparams = F.relu(self.lin2(transparams))
        transparams = self.drop(transparams)
        transparams = self.lin3(transparams)
        transparams = torch.tanh(transparams)

        identitymatrix = torch.eye(2, 3).to(noise.device)
        identitymatrix = identitymatrix.unsqueeze(0)
        identitymatrix = identitymatrix.repeat(noise.shape[0], 1, 1)
        theta = transparams[:, 0:6].view(-1, 2, 3)
        affinematrix = theta + identitymatrix

        colorparams = transparams[:, 6:]

        return affinematrix, colorparams, transparams

    def forward(self, x):
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
        # noise
        bs = x.shape[0]
        self.uniform = Uniform(low=-torch.ones(bs, self.nz).to(x.device), high=torch.ones(bs, self.nz).to(x.device))
        noise = self.uniform.rsample()
        # get transformation parameters
        affinematrix, colorparams, transparams = self.get_transformation_parameters(noise)
        # bring images back to [-1:1]
        x = x * self.std.view(1, 3, 1, 1)
        # compute affine transformation grid
        grid = F.affine_grid(affinematrix, x.size(), align_corners=True)
        # apply affine transformation
        x = F.grid_sample(x, grid, align_corners=True)
        # Change image to [0:1]
        x = x + self.mean.view(1, 3, 1, 1)
         # color transformations
        nb_transform = 4
        transform_order = torch.randperm(nb_transform).to(x.device)
        for i in range(nb_transform):
            if transform_order[i] == 0:
                x = adjust_brightness(x, 1 + colorparams[:, 0].squeeze(-1))
            elif transform_order[i] == 1:
                x = adjust_contrast(x, colorparams[:, 1].squeeze(-1))
            elif transform_order[i] == 2:
                x = adjust_saturation(x, 1 + colorparams[:, 2].squeeze(-1))
            elif transform_order[i] == 3:
                x = adjust_hue(x, colorparams[:, 3].squeeze(-1) * 0.5)
        # restandardize images
        x = (x - self.mean.view(1, 3, 1, 1)) / self.std.view(1, 3, 1, 1)

        # if self.buffer_in.size()[0] == 0:
        #     self.buffer_in = smp.clone().detach()
        # else:
        #     self.buffer_in = torch.cat((self.buffer_in, smp.clone().detach()))
        # if self.buffer_out.size()[0] == 0:
        #     self.buffer_out = transparams.clone().detach()
        # else:
        #     self.buffer_out = torch.cat((self.buffer_out, transparams.clone().detach()))

        transformations = torch.mean(transparams.clone().detach(), dim=0, keepdim=True)

        return x, transformations