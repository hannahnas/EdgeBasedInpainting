import torch.nn as nn
import torch
from modules.convs import GatedConv2d, BasicDeformConv2d
from modules.blocks import GatedDeformableResNetBlockEnc, ResNetBlockEnc, GatedResNetBlockEnc, DeformableResNetBlockEnc


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels, type='base'):
        super().__init__()
        self._create_network(in_channels, type)
        self._init_params()

    def _create_network(self, in_channels, type):

        if type == 'gated':
            resblock = GatedResNetBlockEnc
            first_conv = GatedConv2d
        elif type ==  'deformable':
            resblock = DeformableResNetBlockEnc
            first_conv = BasicDeformConv2d
        elif type == 'gated deformable':
            resblock = GatedDeformableResNetBlockEnc
            first_conv = GatedConv2d
        elif type == 'base':
            resblock = ResNetBlockEnc
            first_conv = nn.Conv2d
        else:
            print('Encoder type does not exist.')

        self.layer_0 = nn.Sequential(
            first_conv(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer_1 = nn.Sequential(
            resblock(64, 64, subsample=False),
            resblock(64, 64, subsample=False)
        )

        self.layer_2 = nn.Sequential(
            resblock(64, 128, subsample=True),
            resblock(128, 128, subsample=False)
        )

        self.layer_3 = nn.Sequential(
            resblock(128, 256, subsample=True),
            resblock(256, 256, subsample=False)
        )

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

        return x


if __name__ == '__main__':
    encoder = ResNetEncoder(in_channels=3)

    x = torch.rand((4, 3, 256, 256))
    out = encoder(x)
    print(out.shape)

