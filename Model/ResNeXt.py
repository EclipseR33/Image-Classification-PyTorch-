import torch
import torch.nn as nn


def conv1x1(in_ch, out_ch, stride=1, padding=0, groups=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride,
                                   padding=padding, groups=groups, bias=False),
                         nn.BatchNorm2d(out_ch))


def conv3x3(in_ch, out_ch, stride=1, padding=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                                   padding=padding, groups=groups, bias=False),
                         nn.BatchNorm2d(out_ch))


class ResNeXtBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, cardinality, downsample=None, stride=1):
        super(ResNeXtBlock, self).__init__()

        self.conv1 = conv1x1(in_ch, mid_ch)
        self.conv2 = conv3x3(mid_ch, mid_ch, stride=stride, groups=cardinality)
        self.conv3 = conv1x1(mid_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.relu(out)

        out += identity
        return out


class ResNeXt(nn.Module):
    def __init__(self, block, blocks_num, num_classes, cardinality, include_top=True, zero_init_residual=False):
        super(ResNeXt, self).__init__()
        self.include_top = include_top

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, blocks_num[0], 64, 256, cardinality, mid_ch=128)
        self.layer2 = self.make_layer(block, blocks_num[1], 256, 512, cardinality, stride=2)
        self.layer3 = self.make_layer(block, blocks_num[2], 512, 1024, cardinality, stride=2)
        self.layer4 = self.make_layer(block, blocks_num[3], 1024, 2048, cardinality, stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNeXtBlock):
                    nn.init.constant_(m.conv3[1].weight, 0)

    def make_layer(self, block, block_num, in_ch, out_ch, cardinality, mid_ch=None, stride=1):
        downsample = None
        if stride != 1 or mid_ch is not None:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        if mid_ch is None:
            mid_ch = in_ch
        layers = [block(in_ch, mid_ch, out_ch, cardinality, downsample, stride=stride)]

        for _ in range(1, block_num):
            layers.append(block(out_ch, mid_ch, out_ch, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnext50_32x4d(num_classes, include_top=True, zero_init_residual=False):
    return ResNeXt(ResNeXtBlock, [3, 4, 6, 3], num_classes, cardinality=32,
                   include_top=include_top, zero_init_residual=zero_init_residual)


def resnext101_32x4d(num_classes, include_top=True, zero_init_residual=False):
    return ResNeXt(ResNeXtBlock, [3, 4, 23, 3], num_classes, cardinality=32,
                   include_top=include_top, zero_init_residual=zero_init_residual)


def resnext50(num_classes, include_top=True, zero_init_residual=False):
    return ResNeXt(ResNeXtBlock, [3, 4, 6, 3], num_classes, cardinality=1,
                   include_top=include_top, zero_init_residual=zero_init_residual)


def resnext101(num_classes, include_top=True, zero_init_residual=False):
    return ResNeXt(ResNeXtBlock, [3, 4, 23, 3], num_classes, cardinality=1,
                   include_top=include_top, zero_init_residual=zero_init_residual)


if __name__ == '__main__':
    from torchvision.models import resnet

    x = torch.randn((2, 3, 224, 224))
    net = resnext50_32x4d(1000, zero_init_residual=True)
    print(net)
    out = net(x)
    print(out.size())
