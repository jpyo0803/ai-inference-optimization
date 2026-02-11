import torch
import torch.nn as nn

'''
    Basic Block
'''
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                        padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride) # padding=1
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) # Save memory
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample # Only used for 2nd, 3rd Basic blocks

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 2nd, 3rd Basic blocks need downsampling as their conv1 has a stride of 2
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = out + identity
        out = self.relu(out)

        return out


class ResNet_CIFAR(nn.Module):
    def __init__(self, depth=20, num_classes=10):
        super().__init__()

        # First convolution and last Linear layer is not counted
        num_blocks = (depth - 2) // 6
        assert (depth - 2) % 6 == 0

        self.in_channels = 16

        # CIFAR input's dimension: (3, 32, 32)
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # 3 stages
        self.layer1 = self._make_layer(16, num_blocks, stride=1) # input dim: (16, 32, 32)
        self.layer2 = self._make_layer(32, num_blocks, stride=2, need_downsample=True) # input dim: (32, 16, 16)
        self.layer3 = self._make_layer(64, num_blocks, stride=2, need_downsample=True) # input dim: (64, 8, 8)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride, need_downsample=False):
        layers = []

        downsample = None
        if need_downsample == True:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.global_avg_pool(out)
        out = torch.flatten(out, 1) # keep batch dimension
        logits = self.fc(out) 
        return logits