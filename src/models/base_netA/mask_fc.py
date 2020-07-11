import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform

class maskFC(nn.Module):
    def __init__(self, imageSize, nc, nz, datasetmean, datasetstd):
        super(maskFC, self).__init__()
        self.imageSize = imageSize
        self.nc = nc
        self.nz = nz
        self.mean = torch.tensor(datasetmean)
        self.std = torch.tensor(datasetstd)
        self.lin1 = nn.Linear(self.nz, self.imageSize * self.imageSize * nc)
        self.lin2 = nn.Linear(self.imageSize * self.imageSize * nc, self.imageSize * self.imageSize * nc)
        self.drop = nn.Dropout(0.2)
        self.buffer_in = torch.Tensor()
        self.buffer_out = torch.Tensor()

    def get_mask(self, noise):
        mask = self.lin1(noise)
        mask = self.drop(mask)
        mask = self.lin2(mask)
        mask = mask.view(noise.size(0), self.nc, self.imageSize, self.imageSize)
        mask = torch.tanh(mask)

        return mask
    
    def forward(self, x):
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)        
        # noise
        bs = x.shape[0]
        self.uniform = Uniform(low=-torch.ones(bs, self.nz).to(x.device), high=torch.ones(bs, self.nz).to(x.device))
        noise = self.uniform.rsample()
        # get transformation        
        mask = self.get_mask(noise)
        # Bring images back to [-1:1]
        x = x * self.std.view(1, 3, 1, 1)
        # apply transformation
        x = torch.clamp(mask + x, min=-1, max=1) 
        # restandardize images and masks
        x = x / self.std.view(1, 3, 1, 1)
        masks = mask / self.std.view(1, 3, 1, 1)

        return x, masks, self.buffer_out