import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform

class maskCNN(nn.Module):
    def __init__(self, imageSize, nc, nz, datasetmean, datasetstd):
        super(maskCNN, self).__init__()
        self.imageSize = imageSize
        self.nc = nc
        self.nz = nz
        self.mean = torch.tensor(datasetmean)
        self.std = torch.tensor(datasetstd)
        self.deconv1 = nn.ConvTranspose2d(self.nz, 8, 8, 1, 0, bias=False)
        self.deconv2 = nn.ConvTranspose2d(8, 16, 2, 2, 0, bias=False)
        self.deconv3 = nn.ConvTranspose2d(16, 32, 2, 2, 0, bias=False)
        self.deconv4 = nn.ConvTranspose2d(32, self.nc, 1, 1, 0, bias=False)
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

    def forward(self, x):
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)        
        #noise
        bs = x.shape[0]
        self.uniform = Uniform(low=-torch.ones(bs, self.nz).to(x.device), high=torch.ones(bs, self.nz).to(x.device))
        noise = self.uniform.rsample()
        # compute transformation
        mask = self.get_mask(noise)
        # Bring images back to [-1:1]
        x = x * self.std.view(1, 3, 1, 1)
        # apply transformation
        x = torch.clamp(x + mask, min=-1, max=1) 
        # restandardize images and masks
        x = x / self.std.view(1, 3, 1, 1)
        masks = mask / self.std.view(1, 3, 1, 1)

        return x, masks, self.buffer_out