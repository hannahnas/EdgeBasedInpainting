import torch.nn.functional as F
import torch.nn as nn
from torchvision.ops import DeformConv2d


class GatedConv2d(nn.Module):
    """
    Gated Convlution layer with activation
    Params: same as conv2d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(GatedConv2d, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=True)
        self.sigmoid = nn.Sigmoid()

    def gated(self, mask):
        # return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        x = x * self.gated(mask)

        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, scale=2, padding=1, mode='bilinear', bias=False, batch_norm=True):
        super().__init__()
        self.scale = scale
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=1, padding=padding, bias=bias)
        self.batch_norm = batch_norm
        self.batchnorm2d = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.mode == 'bilinear':
            x = F.interpolate(x, scale_factor=self.scale,
                              mode=self.mode, align_corners=True)
        else:
            x = F.interpolate(x, scale_factor=self.scale,
                              mode=self.mode)
        x = self.conv(x)

        if self.batch_norm:
            x = self.batchnorm2d(x)

        return x

class BasicDeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=1, bias=False, dilation=1, groups=1, offset_groups=1):
        super().__init__()
        offset_channels = 2 * kernel_size * kernel_size
        self.conv2d_offset = nn.Conv2d(
            in_channels,
            offset_channels * offset_groups,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2d = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
    
    def forward(self, x):
        offset = self.conv2d_offset(x)
        return self.conv2d(x, offset)