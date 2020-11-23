import math

import torch.nn as nn
import torch.nn.functional as F

from models import base_3d, aspp_3d


class FlatDeeplab(nn.Module):

    def seed(self, x):
        # x = self.relu(self.bn1(self.conv1(x))) -- CIFAR
        # x = self.maxpool(self.relu(self.bn1(self.conv1(x)))) -- ImageNet
        raise NotImplementedError

    # run a variable policy batch through the resnet implemented as a full mask over the residual
    # fast to train, non-indicative of time saving (use forward_single instead)
    def forward(self, x, policy=None, fully=False):
        if fully is True:
            return self.forward_full(x)
        else:
            h, w, z = x.size()[2], x.size()[3], x.size()[4]

            x = self.seed(x)

            t = 0
            for segment, num_blocks in enumerate(self.layer_config):
                for b in range(num_blocks):
                    action = policy[:, t].contiguous()
                    residual = self.ds[segment](x) if b == 0 else x

                    # early termination if all actions in the batch are zero
                    if action.data.sum() == 0:
                        x = residual
                        t += 1
                        continue

                    action_mask = action.float().view(-1, 1, 1, 1, 1)
                    fx = F.relu(residual + self.blocks[segment][b](x))
                    x = fx * action_mask + residual * (1 - action_mask)
                    t += 1

            x = self.conv_seg(x)
            x = F.interpolate(x, size=(h, w, z), mode="trilinear", align_corners=True)
            return x

    # run a single, fixed policy for all items in the batch
    # policy is a (15,) vector. Use with batch_size=1 for profiling
    def forward_single(self, x, policy):
        h, w, z = x.size()[2], x.size()[3], x.size()[4]
        x = self.seed(x)

        t = 0
        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                residual = self.ds[segment](x) if b == 0 else x
                if policy[t] == 1:
                    x = residual + self.blocks[segment][b](x)
                    x = F.relu(x)
                else:
                    x = residual
                t += 1

        x = self.conv_seg(x)
        x = F.interpolate(x, size=(h, w, z), mode="trilinear", align_corners=True)
        return x

    def forward_feature(self, x):
        x = self.seed(x)

        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                residual = self.ds[segment](x) if b == 0 else x
                x = F.relu(residual + self.blocks[segment][b](x))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward_full(self, x):
        h, w, z = x.size()[2], x.size()[3], x.size()[4]
        x = self.seed(x)

        for segment, num_blocks in enumerate(self.layer_config):
            for b in range(num_blocks):
                residual = self.ds[segment](x) if b == 0 else x
                x = F.relu(residual + self.blocks[segment][b](x))

        x = self.conv_seg(x)
        x = F.interpolate(x, size=(h, w, z), mode="trilinear", align_corners=True)
        return x


# Smaller Flattened DeeplabV3, tailored for small size image
class FlatDeeplab32(FlatDeeplab):

    def __init__(self, block, layers, num_classes=10):
        super(FlatDeeplab32, self).__init__()

        self.inplanes = 16
        self.conv1 = base_3d.conv3x3(1, 16)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool3d((8, 8, 4))

        strides = [1, 2, 2]
        filt_sizes = [16, 32, 64]
        self.blocks, self.ds = [], []
        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)
        self.conv_seg = nn.Sequential(
            aspp_3d.ASPP(512 * block.expansion),
            nn.ConvTranspose3d(
                256,
                32,
                2,
                stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                32,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                32,
                num_classes,
                kernel_size=1,
                stride=1,
                bias=False)
        )

        self.layer_config = layers

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def seed(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return x

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = nn.Sequential()
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = base_3d.DownsampleB(self.inplanes, planes * block.expansion, stride)

        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1))

        return layers, downsample


# Regular Flattened DeeplabV3, tailored for large size image.
class FlatDeeplab224(FlatDeeplab):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(FlatDeeplab224, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        strides = [1, 2, 2, 2]
        filt_sizes = [64, 128, 256, 512]
        self.blocks, self.ds = [], []
        for idx, (filt_size, num_blocks, stride) in enumerate(zip(filt_sizes, layers, strides)):
            blocks, ds = self._make_layer(block, filt_size, num_blocks, stride=stride)
            self.blocks.append(nn.ModuleList(blocks))
            self.ds.append(ds)

        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)
        self.avgpool = nn.AvgPool3d((4, 4, 6))

        self.conv_seg = nn.Sequential(
            aspp_3d.ASPP(512 * block.expansion),
            nn.ConvTranspose3d(
                256,
                32,
                2,
                stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                32,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                32,
                num_classes,
                kernel_size=1,
                stride=1,
                bias=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.layer_config = layers

    def seed(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return x

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = nn.Sequential()
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers, downsample


class Policy32(nn.Module):

    def __init__(self, layer_config=[1, 1, 1], num_blocks=15):
        super(Policy32, self).__init__()
        self.features = FlatDeeplab32(base_3d.BasicBlock, layer_config, num_classes=1)
        self.feat_dim = self.features.conv_seg[0].inplanes
        self.features.conv_seg = nn.Sequential()

        self.logit = nn.Linear(self.feat_dim, num_blocks)
        self.vnet = nn.Linear(self.feat_dim, 1)

    def load_state_dict(self, state_dict):
        # support legacy models
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('features.fc')}
        return super(Policy32, self).load_state_dict(state_dict)

    def forward(self, x):
        x = self.features.forward_feature(x)
        value = self.vnet(x)
        probs = F.sigmoid(self.logit(x))
        return probs, value


class Policy224(nn.Module):

    def __init__(self, layer_config=[1, 1, 1, 1], num_blocks=16):
        super(Policy224, self).__init__()
        self.features = FlatDeeplab224(base_3d.BasicBlock, layer_config, num_classes=1)

        self.features.avgpool = nn.AvgPool3d(4)
        self.feat_dim = self.features.conv_seg[0].inplanes
        self.features.conv_seg = nn.Sequential()

        self.logit = nn.Linear(self.feat_dim, num_blocks)
        self.vnet = nn.Linear(self.feat_dim, 1)

    def load_state_dict(self, state_dict):
        # support legacy models
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('features.fc')}
        return super(Policy224, self).load_state_dict(state_dict)

    def forward(self, x):
        x = F.avg_pool3d(x, 2)
        x = self.features.forward_feature(x)
        value = self.vnet(x)
        probs = F.sigmoid(self.logit(x))
        return probs, value
