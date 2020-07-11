# Source https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)
from torchmeta.modules.utils import get_subdict

class BasicBlock(MetaModule):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = MetaConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)

        self.shortcut = MetaSequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = MetaSequential(
                MetaConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                MetaBatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, params=None):
        out = F.relu(self.bn1(self.conv1(x, params=get_subdict(params, 'conv1')), params=get_subdict(params, 'bn1')))
        out = self.bn2(self.conv2(out, params=get_subdict(params, 'conv2')), params=get_subdict(params, 'bn2'))
        out += self.shortcut(x, params=get_subdict(params, 'shortcut'))
        out = F.relu(out)
        return out


class Bottleneck(MetaModule):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = MetaConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)
        self.conv3 = MetaConv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = MetaBatchNorm2d(self.expansion*planes)

        self.shortcut = MetaSequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = MetaSequential(
                MetaConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                MetaBatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, params=None):
        out = F.relu(self.bn1(self.conv1(x, params=get_subdict(params, 'conv1')), params=get_subdict(params, 'bn1')))
        out = F.relu(self.bn2(self.conv2(out, params=get_subdict(params, 'conv2')), params=get_subdict(params, 'bn2')))
        out = self.bn3(self.conv3(out, params=get_subdict(params, 'conv3')), params=get_subdict(params, 'bn3'))
        out += self.shortcut(x, params=get_subdict(params, 'shortcut'))
        out = F.relu(out)
        return out


class ResNet(MetaModule):
    def __init__(self, nc, num_classes, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = MetaConv2d(nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = MetaLinear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return MetaSequential(*layers)

    def forward(self, x, params=None):
        out = F.relu(self.bn1(self.conv1(x, params=get_subdict(params, 'conv1')), params=get_subdict(params, 'bn1')))
        out = self.layer1(out, params=get_subdict(params, 'layer1'))
        out = self.layer2(out, params=get_subdict(params, 'layer2'))
        out = self.layer3(out, params=get_subdict(params, 'layer3'))
        out = self.layer4(out, params=get_subdict(params, 'layer4'))
        out = F.avg_pool2d(out, out.size()[2:])
        out = out.view(out.size(0), -1)
        out = self.linear(out, params=get_subdict(params, 'linear'))
        return out


def ResNet18(nc, nclasses):
    return ResNet(nc, nclasses, BasicBlock, [2, 2, 2, 2],)

def ResNet34(nc, nclasses):
    return ResNet(nc, nclasses, BasicBlock, [3, 4, 6, 3])

def ResNet50(nc, nclasses):
    return ResNet(nc, nclasses, Bottleneck, [3, 4, 6, 3])

def ResNet101(nc, num_classes):
    return ResNet(nc, Bottleneck, [3, 4, 23, 3])

def ResNet152(nc, num_classes):
    return ResNet(nc, Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()