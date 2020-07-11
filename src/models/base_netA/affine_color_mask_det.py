import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform
from .color_utils import *
import kornia

# For transformations with discrete parameters
from .det_activations import DeterministicBinaryActivation, StochasticBinaryActivation

class affineColorMaskDet(nn.Module):
    def __init__(self, imageSize, nc, nz, datasetmean, datasetstd, neurons=10, mode='Deterministic', estimator='ST'):
        super(affineColorMaskDet, self).__init__()

        # For transformations with discrete parameters
        # --------------------------------------------
        assert mode in ['Deterministic', 'Stochastic']
        assert estimator in ['ST', 'REINFORCE']
        self.mode = mode
        self.estimator = estimator
        if self.mode == 'Deterministic':
            self.act = DeterministicBinaryActivation(estimator=estimator)
        elif self.mode == 'Stochastic':
            self.act = StochasticBinaryActivation(estimator=estimator)
        # --------------------------------------------

        self.imageSize = imageSize
        self.nc = nc
        self.nz = nz
        self.mean = torch.tensor(datasetmean)
        self.std = torch.tensor(datasetstd)
        self.lin1 = nn.Linear(self.nz, neurons)
        self.lin2 = nn.Linear(neurons, 10 * neurons)
        self.lin3 = nn.Linear(10*neurons, 11)
        self.deconv1 = nn.ConvTranspose2d(self.nz, int(self.imageSize * 0.25), int(self.imageSize * 0.25), 1, 0, bias=False)
        self.deconv2 = nn.ConvTranspose2d(int(self.imageSize * 0.25), int(self.imageSize * 0.5), 2, 2, 0, bias=False)
        self.deconv3 = nn.ConvTranspose2d(int(self.imageSize * 0.5), self.imageSize, 2, 2, 0, bias=False)
        self.deconv4 = nn.ConvTranspose2d(self.imageSize, self.nc, 1, 1, 0, bias=False)
        self.drop = nn.Dropout(0.2)
        self.buffer_in = torch.Tensor()
        self.buffer_out = torch.Tensor()

    def get_mask(self, noise):
        mask = self.deconv1(noise.unsqueeze(-1).unsqueeze(-1))
        mask = self.drop(mask)
        mask = self.deconv2(mask)
        mask = self.drop(mask)
        mask = self.deconv3(mask)
        mask = self.drop(mask)
        mask = self.deconv4(mask)
        mask = torch.tanh(mask)
        
        return mask

    def get_transformation_parameters(self, noise, slope):
        transparams = F.relu(self.lin1(noise))
        transparams = self.drop(transparams)
        transparams = F.relu(self.lin2(transparams))
        transparams = self.drop(transparams)
        transparams = self.lin3(transparams)
        transparams[:, 0:10] = torch.tanh(transparams[:, 0:10])
        transparams[:, 10] = self.act((transparams[:, 10], slope))

        identitymatrix = torch.eye(2, 3).to(noise.device)
        identitymatrix = identitymatrix.unsqueeze(0)
        identitymatrix = identitymatrix.repeat(noise.shape[0], 1, 1)
        theta = transparams[:, 0:6].view(-1, 2, 3)
        affinematrix = theta + identitymatrix

        colorparams = transparams[:, 6:10]

        flipparam = transparams[:, 10]

        return affinematrix, colorparams, flipparam, transparams

    def forward(self, x, slope):
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)        
        # noise
        bs = x.shape[0]
        self.uniform = Uniform(low=-torch.ones(bs, self.nz).to(x.device), high=torch.ones(bs, self.nz).to(x.device))
        noise = self.uniform.rsample()
        # get transformation parameters
        affinematrix, colorparams, flipparam, transparams = self.get_transformation_parameters(noise, slope)
        # apply flip transformation
        x = (flipparam.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x) * hflip(x)) + ((1 - flipparam.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) * x)

        # bring images back to [-1:1]
        x = x * self.std.view(1, 3, 1, 1)
        # compute affine transformation grid
        grid = F.affine_grid(affinematrix, x.size(), align_corners=True)
        # apply affine transformations
        x = F.grid_sample(x, grid, align_corners=True)
        # bring images back to [0:1]
        x = x + self.mean.view(1, 3, 1, 1)
        # apply color transformations
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

        self.uniform2= Uniform(low=-torch.ones(bs, self.nz).to(x.device), high=torch.ones(bs, self.nz).to(x.device))
        noise2 = self.uniform2.rsample()
        mask = self.get_mask(noise2)
        # change image to [-1:1]
        x = x - self.mean.view(1, 3, 1, 1)
        # apply mask and restandardize images
        x = torch.clamp(mask + x, min=-1, max=1) / self.std.view(1, 3, 1, 1)

        # if self.buffer_in.size()[0] == 0:
        #     self.buffer_in = smp.clone().detach()
        # else:
        #     self.buffer_in = torch.cat((self.buffer_in, smp.clone().detach()))
        if self.buffer_out.size()[0] == 0:
            self.buffer_out = transparams.clone().detach()
        else:
            self.buffer_out = torch.cat((self.buffer_out, transparams.clone().detach()))

        return x, self.buffer_out
