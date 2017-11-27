import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, height, width, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        # SE layers
        self.fc_c1 = nn.Conv2d(planes, planes//16, kernel_size=1) # use Conv2d instead of linear layer
        self.fc_c2 = nn.Conv2d(planes//16, planes, kernel_size=1)
        # SE for height and width
        mid_height = int(np.maximum(height//4, 1))
        mid_width = int(np.maximum(width//4, 1))
        self.fc_h1 = nn.Conv2d(height, mid_height, kernel_size=1)
        self.fc_h2 = nn.Conv2d(mid_height, height, kernel_size=1)
        self.fc_w1 = nn.Conv2d(width, mid_width, kernel_size=1)
        self.fc_w2 = nn.Conv2d(mid_width, width, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # SE for channels
        c = F.avg_pool2d(out, out.size(2))
        c = F.relu(self.fc_c1(c))
        c = F.sigmoid(self.fc_c2(c))
        # out = out * c  # broadcasting

        num, channel, height, width = out.size()
        # SE for height
        h_out = out.permute(0, 2, 1, 3)
        h = F.avg_pool2d(h_out, (channel, width))
        h = F.relu(self.fc_h1(h))
        h = F.sigmoid(self.fc_h2(h))
        h = h.permute(0, 2, 1, 3)

        # SE for width
        w_out = out.permute(0, 3, 2, 1)
        w = F.avg_pool2d(w_out, (height, channel))
        w = F.relu(self.fc_h1(w))
        w = F.sigmoid(self.fc_h2(w))
        w = w.permute(0, 3, 2, 1)

        # broadcasting
        out = out * h * w * c

        out += self.shortcut(x)
        out = F.relu(out)
        return out

class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, height, width, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        height = int(height/stride)
        width = int(width/stride)

        # SE layers
        self.fc_c1 = nn.Conv2d(planes, planes//16, kernel_size=1) # use Conv2d instead of linear layer
        self.fc_c2 = nn.Conv2d(planes//16, planes, kernel_size=1)
        # SE for height and width
        mid_height = int(np.maximum(height//4, 1))
        mid_width = int(np.maximum(width//4, 1))
        self.fc_h1 = nn.Conv2d(height, mid_height, kernel_size=1)
        self.fc_h2 = nn.Conv2d(mid_height, height, kernel_size=1)
        self.fc_w1 = nn.Conv2d(width, mid_width, kernel_size=1)
        self.fc_w2 = nn.Conv2d(mid_width, width, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # SE for channels
        c = F.avg_pool2d(out, out.size(2))
        c = F.relu(self.fc_c1(c))
        c = F.sigmoid(self.fc_c2(c))
        # out = out * c  # broadcasting

        num, channel, height, width = out.size()
        # SE for height
        h_out = out.permute(0, 2, 1, 3)
        h = F.avg_pool2d(h_out, (channel, width))
        h = F.relu(self.fc_h1(h))
        h = F.sigmoid(self.fc_h2(h))
        h = h.permute(0, 2, 1, 3)

        # SE for width
        w_out = out.permute(0, 3, 2, 1)
        w = F.avg_pool2d(w_out, (height, channel))
        w = F.relu(self.fc_h1(w))
        w = F.sigmoid(self.fc_h2(w))
        w = w.permute(0, 3, 2, 1)

        # broadcasting
        out = out * h * w * c

        out += shortcut
        return out


class ATTNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ATTNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], 32, 32, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 32, 32, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 16, 16, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 8,  8,   stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, height, width, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, height, width, stride))
            height = int(height/stride)
            width = int(width/stride)
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ATTNet18():
    return ATTNet(PreActBlock, [2,2,2,2])


def test():
    net = ATTNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()





























