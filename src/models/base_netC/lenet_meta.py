'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)
from torchmeta.modules.utils import get_subdict

class LeNet(MetaModule):
    def __init__(self,nc, ncat):
        super(LeNet, self).__init__()
        self.conv1 = MetaConv2d(nc, 6, 5)
        self.conv2 = MetaConv2d(6, 16, 5)
        self.fc1   = MetaLinear(16*5*5, 120)
        self.fc2   = MetaLinear(120, 84)
        self.fc3   = MetaLinear(84, ncat)

    def forward(self, x, params=None):
        out = F.relu(self.conv1(x, params=get_subdict(params, 'conv1')))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out, params=get_subdict(params, 'conv2')))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out, params=get_subdict(params, 'fc1')))
        out = F.relu(self.fc2(out, params=get_subdict(params, 'fc2')))
        out = self.fc3(out)
        return out










