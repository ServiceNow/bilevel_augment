import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform


class randomAugmenter(nn.Module):
    def __init__(self, c, transformation, datasetmean, datasetstd):
        super(randomAugmenter, self).__init__()
        self.c = float(c)
        self.transformation = transformation
        self.mean = torch.tensor(datasetmean)
        self.std = torch.tensor(datasetstd)
        self.buffer_in = torch.Tensor()
        self.buffer_out = torch.Tensor()

    def get_affine_matrix(self, noise, transform):
        identitymatrix = torch.eye(2, 3).to(noise.device)
        identitymatrix = identitymatrix.unsqueeze(0)
        identitymatrix = identitymatrix.repeat(noise.shape[0], 1, 1)
        if self.transformation == 'translation':
            affinematrix = identitymatrix
            affinematrix[:, :, 2] = noise
        elif self.transformation == 'scale':
            affinematrix = identitymatrix
            affinematrix[:, 0, 0] += noise[:, 0]
            affinematrix[:, 1, 1] += noise[:, 1]
        elif self.transformation == 'rotation':
            affinematrix = identitymatrix
            affinematrix[:, 0, 0:1] += noise[:, 0:1]
            affinematrix[:, 1, 0:1] += noise[:, 2:3]
        else:
            affinematrix = identitymatrix + noise.view(-1, 2, 3)

        return affinematrix

    def forward(self, x):
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)        
        # noise
        bs = x.shape[0]
        if self.transformation in ['translation', 'scale']:
            self.uniform = Uniform(low=-self.c * torch.ones(bs, 2).to(x.device), high=self.c * torch.ones(bs, 2).to(x.device))
        elif self.transformation in ['rotation']:
            self.uniform = Uniform(low=-self.c * torch.ones(bs, 4).to(x.device), high=self.c * torch.ones(bs, 4).to(x.device))
        else:
            self.uniform = Uniform(low=-self.c * torch.ones(bs, 6).to(x.device), high=self.c * torch.ones(bs, 6).to(x.device))
        noise = self.uniform.rsample()
        # get transformation matrix
        affinematrix = self.get_affine_matrix(noise, self.transformation)
        # compute transformation grid        
        grid = F.affine_grid(affinematrix, x.size(), align_corners=True)
        # Bring back images to [-1;1]
        x = (x * self.std.view(1, 3, 1, 1)) # + self.mean.view(1, 3, 1, 1)
        # apply transformation
        x = F.grid_sample(x, grid, align_corners=True)
        # Restandardize image
        x = x / self.std.view(1, 3, 1, 1)

        if self.buffer_in.size()[0] == 0:
            self.buffer_in = noise.clone().detach()
        else:
            self.buffer_in = torch.cat((self.buffer_in, noise.clone().detach()))
        if self.buffer_out.size()[0] == 0:
            self.buffer_out = affinematrix.clone().detach()
        else:
            self.buffer_out = torch.cat((self.buffer_out, affinematrix.clone().detach()))

        return x, self.buffer_out