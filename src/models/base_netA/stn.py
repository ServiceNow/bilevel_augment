import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform


class STN(nn.Module):
    def __init__(self, isize, n_channels,  n_filters, nz, datasetmean, datasetstd):
        super(STN, self).__init__()
        self.nz = nz
        self.mean = torch.tensor(datasetmean)
        self.std = torch.tensor(datasetstd)
        self.buffer_in = torch.Tensor()
        self.buffer_out = torch.Tensor()
        self.project = nn.ConvTranspose2d(self.nz, int(n_filters * 0.5), isize, 1, 0, bias=True)
        self.inconv = nn.Conv2d(n_channels, int(n_filters * 0.5), 3, padding=1)
        self.localization = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 32),
            nn.ReLU(),
            nn.Linear(32, 3 * 2)
        )

    def get_affine_matrix(self, input):
        xs = self.localization(input)
        xs = xs.view(-1, 4096)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        identitymatrix = torch.eye(2, 3).to(input.device)
        identitymatrix = identitymatrix.unsqueeze(0)
        identitymatrix = identitymatrix.repeat(input.shape[0], 1, 1)
        affinematrix = identitymatrix + theta
  
        return affinematrix

    def forward(self, x):
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)        
        # input image
        xconv = self.inconv(x)
        # input noise
        bs = x.shape[0]
        self.uniform = Uniform(low=-torch.ones(bs, self.nz).to(x.device), high=torch.ones(bs, self.nz).to(x.device))
        z = self.uniform.rsample()
        noise = self.project(z.unsqueeze(-1).unsqueeze(-1))
        # concat and get transformation matrix
        input = torch.cat([xconv, noise], dim=1)
        affinematrix = self.get_affine_matrix(input)
        
        # compute transformation grid
        grid = F.affine_grid(affinematrix, x.size(), align_corners=True)
        # Bring back images to [-1;1]
        x = (x * self.std.view(1, 3, 1, 1))
        # apply transformation
        x = F.grid_sample(x, grid, align_corners=True)
        # Restandardize image
        x = x / self.std.view(1, 3, 1, 1)

        if self.buffer_in.size()[0] == 0:
            self.buffer_in = z.clone().detach()
        else:
            self.buffer_in = torch.cat((self.buffer_in, z.clone().detach()))
        if self.buffer_out.size()[0] == 0:
            self.buffer_out = affinematrix.clone().detach()
        else:
            self.buffer_out = torch.cat((self.buffer_out, affinematrix.clone().detach()))

        return x, self.buffer_out