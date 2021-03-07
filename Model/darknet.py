import torch
import torch.nn as nn


def conv(in_ch, out_ch, k_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1)
    )


class Block(nn.Module):
    def __init__(self, in_ch):
        super(Block, self).__init__()
        mid_ch = int(in_ch / 2)
        self.conv1 = conv(in_ch, mid_ch, k_size=1, padding=0)
        self.conv2 = conv(mid_ch, in_ch)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += identity
        return out


def make_layer(block, num_block, in_ch):
    layers = []
    for _ in range(num_block):
        layers.append(block(in_ch))
    return nn.Sequential(*layers)


class DarkNet(nn.Module):
    def __init__(self, block, num_blocks, in_ch=3):
        super(DarkNet, self).__init__()

        self.conv0 = conv(in_ch, 32)

        self.conv1 = conv(32, 64, stride=2)
        self.layer1 = make_layer(block, num_blocks[0], 64)

        self.conv2 = conv(64, 128, stride=2)
        self.layer2 = make_layer(block, num_blocks[1], 128)

        self.conv3 = conv(128, 256, stride=2)
        self.layer3 = make_layer(block, num_blocks[2], 256)

        self.conv4 = conv(256, 512, stride=2)
        self.layer4 = make_layer(block, num_blocks[3], 512)

        self.conv5 = conv(512, 1024, stride=2)
        self.layer5 = make_layer(block, num_blocks[4], 1024)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv0(x)

        x = self.conv1(x)
        x = self.layer1(x)

        x = self.conv2(x)
        x = self.layer2(x)

        x3 = self.conv3(x)
        x3 = self.layer3(x3)

        x4 = self.conv4(x3)
        x4 = self.layer4(x4)

        x5 = self.conv5(x4)
        x5 = self.layer5(x5)
        return x3, x4, x5


def darknet(num_blocks=[1, 2, 8, 8, 4]):
    net = DarkNet(Block, num_blocks)
    return net


def darknet_with_fc(num_classes, num_blocks=[1, 2, 8, 8, 4]):
    net = nn.Sequential(
        DarkNet(Block, num_blocks),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(1024, num_classes)
    )
    return net


if __name__ == '__main__':
    net = darknet_with_fc(1000)

    print(net)
    x = torch.randn((1, 3, 512, 512))
    out = net(x)

    print(out.size())
