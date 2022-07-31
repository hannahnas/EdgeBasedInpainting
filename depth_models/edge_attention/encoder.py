from modules.convs import GatedConv2d
from modules.blocks import ResNetBlockEnc
import torch
import torch.nn as nn


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels, gated=False):
        super().__init__()
        self._create_network(in_channels, gated=gated)
        self._init_params()

    def _create_network(self, in_channels, gated):

        if gated:
            self.layer_0 = nn.Sequential(
                GatedConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        else:
            self.layer_0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

        self.layer_1 = nn.Sequential(
            ResNetBlockEnc(64, 64, subsample=False, gated=gated),
            ResNetBlockEnc(64, 64, subsample=False, gated=gated)
        )

        self.layer_2 = nn.Sequential(
            ResNetBlockEnc(64, 128, subsample=True, gated=gated),
            ResNetBlockEnc(128, 128, subsample=False, gated=gated)
        )

        self.layer_3 = nn.Sequential(
            ResNetBlockEnc(128, 256, subsample=True, gated=gated),
            ResNetBlockEnc(256, 256, subsample=False, gated=gated)
        )

        # self.layer_4 = nn.Sequential(
        #     ResNetBlockEnc(256, 512, act_fn, subsample=True,
        #                    deformable=deformable),
        #     ResNetBlockEnc(512, 512, act_fn, subsample=False,
        #                    deformable=deformable)
        # )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        # x = self.layer_4(x)

        return x