import torch
import torch.nn as nn
import torch.nn.functional as F
from Options_all import BaseOptions
opt = BaseOptions().parse()


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):      # (64, 128)
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)            
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))    # channel/2
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))  # channel/4
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))  # chaneel/4

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):      # (-1, 64, 128, 128)
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)     # (-1, 64, 128, 128)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)     # (-1, 32, 128, 128)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)     # (-1, 32, 128, 128)

        out3 = torch.cat((out1, out2, out3), 1)         # 拼接 通道 (-1, 64+32+32, 128, 128) (-1, 128, 128, 128)

        if self.downsample is not None:
            residual = self.downsample(residual)        # (-1, 128, 128, 128)

        out3 += residual

        return out3         # (-1, 128, 128, 128)


class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features):   # (1, 4, 256)
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.dropout = nn.Dropout(0.5)

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(256, 256))

        self.add_module('b2_' + str(level), ConvBlock(256, 256))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))

        self.add_module('b3_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):     # (4, (-1, 256, 64, 64)
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)
        up1 = self.dropout(up1)
        # Lower branch
        low1 = F.max_pool2d(inp, 2, stride=2)       # ( -1, 256, 32, 32)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)
        up1size = up1.size()
        rescale_size = (up1size[2], up1size[3])
        up2 = F.upsample(low3, size=rescale_size, mode='bilinear')

        return up1 + up2

    def forward(self, x):   # (-1, 256, 64, 64)
        return self._forward(self.depth, x)


class FAN_use(nn.Module):

    def __init__(self, config=opt):
        super(FAN_use, self).__init__()
        self.num_modules = 1

        # Base part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)     
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)


        # Stacking part
        hg_module = 0
        self.add_module('m' + str(hg_module), HourGlass(1, 4, 256))
        self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
        self.add_module('conv_last' + str(hg_module),
                        nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
        self.add_module('l' + str(hg_module), nn.Conv2d(256,
                                                        68, kernel_size=1, stride=1, padding=0))
        self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))

        if hg_module < self.num_modules - 1:
            self.add_module(
                'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('al' + str(hg_module), nn.Conv2d(68,
                                                             256, kernel_size=1, stride=1, padding=0))
        # if config.load_pretrain:
        #         self.load_pretrain(config.fan_pretrain_path2)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=5, stride=3, padding=1)
        self.avgpool = nn.MaxPool2d((2, 2), 2)
        self.conv6 = nn.Conv2d(68, 1, 3, 2, 1)
        self.fc = nn.Linear(1024, 256)

    def forward(self, x):           # (-1, 3, 256, 256)
        x = F.relu(self.bn1(self.conv1(x)), True)   # (-1, 64, 128, 128)
        x = F.max_pool2d(self.conv2(x), 2)          # (-1, 128, 128, 128)---->(-1, 128, 64, 64)
        x = self.conv3(x)                           # (-1, 128, 64, 64)
        x = self.conv4(x)                           # (-1, 256, 64, 64)

        previous = x        # (-1, 256, 64, 64)

        i = 0
        hg = self._modules['m' + str(i)](previous)      # (-1, 

        ll = hg
        ll = self._modules['top_m_' + str(i)](ll)

        ll = self._modules['bn_end' + str(i)](self._modules['conv_last' + str(i)](ll))
        tmp_out = self._modules['l' + str(i)](F.relu(ll))
        return tmp_out

    def load_pretrain(self, pretrain_path):
        check_point = torch.load(pretrain_path)
        self.load_state_dict(check_point['check_point'])


class FanFusion(nn.Module):
    def __init__(self, opt=opt):
        super(FanFusion, self).__init__()
        self.opt = opt
        self.model = FAN_use(opt)
        self.conv6 = nn.Conv2d(68, 1, 3, 2, 1)
        self.fc = nn.Linear(1024, 256)
        self.bn1 = nn.BatchNorm2d(68)
        self.relu = nn.ReLU(True)

    def _forward(self, x):          # (-1, 3, 256, 256)
        net = self.model.forward(x)
        net = self.relu(self.bn1(net))
        net = self.conv6(net)
        net = net.view(-1, 1024)
        net = self.relu(net)
        net = self.fc(net)
        return net

    def forward(self, x):
        x0 = x.view(-1, self.opt.image_channel_size, self.opt.image_size, self.opt.image_size)      # (-1, 3, 256, 256)
        net = self._forward(x0)
        net = net.view(x.size(0), -1, 256)
        return net          # (-1, n, 256)
