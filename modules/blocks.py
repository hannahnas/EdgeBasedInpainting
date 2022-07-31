import torch.nn as nn
from modules.convs import GatedConv2d, UpConv, BasicDeformConv2d

class ResNetBlockEnc(nn.Module):

    def __init__(self, c_in, c_out, subsample=False):
        """
        Inputs:
            c_in - Number of input features
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in

        # Network representing F
        self.net = nn.Sequential(
            # No bias needed as the Batch Norm handles it
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1,
                        stride=1 if not subsample else 2, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out)
        )

        # 1x1 convolution with stride 2 means we take the upper left value, and transform it to new output size
        self.downsample = nn.Conv2d(
            c_in, c_out, kernel_size=1, stride=2) if subsample else None
        self.act_fn = nn.ReLU(inplace=True)


    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)

        out = z + x
        out = self.act_fn(out)
        return out

class ResNetBlockDec(nn.Module):

    def __init__(self, c_in, c_out, subsample=False):
        """
        Inputs:
            c_in - Number of input features
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in

            self.net = nn.Sequential(
                # No bias needed as the Batch Norm handles it
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
        else:
            self.net = nn.Sequential(
                # No bias needed as the Batch Norm handles it
                UpConv(c_in, c_out, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out)
            )

        self.upsample = UpConv(c_in, c_out, kernel_size=1, padding=0, batch_norm=False, bias=True) if subsample else None
        self.act_fn = nn.ReLU(inplace=True)

    def forward(self, x):
        z = self.net(x)
        if self.upsample is not None:
            x = self.upsample(x)

        out = z + x
        out = self.act_fn(out)
        return out


class GatedResNetBlockEnc(nn.Module):

    def __init__(self, c_in, c_out, subsample=False):
        """
        Inputs:
            c_in - Number of input features
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in

        # Network representing F
        self.net = nn.Sequential(
            # No bias needed as the Batch Norm handles it
            GatedConv2d(c_in, c_out, kernel_size=3,
                            stride=1 if not subsample else 2),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            GatedConv2d(c_out, c_out, kernel_size=3,
                            stride=1),
            nn.BatchNorm2d(c_out)
        )

        # 1x1 convolution with stride 2 means we take the upper left value, and transform it to new output size
        self.downsample = nn.Conv2d(
            c_in, c_out, kernel_size=1, stride=2) if subsample else None
        self.act_fn = nn.ReLU(inplace=True)


    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        out = self.act_fn(out)
        return out


class DeformableResNetBlockEnc(nn.Module):

    def __init__(self, c_in, c_out, subsample=False):
        """
        Inputs:
            c_in - Number of input features
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in

        # Network representing F

        self.net = nn.Sequential(
            # No bias needed as the Batch Norm handles it
            BasicDeformConv2d(c_in, c_out, kernel_size=3,
                            stride=1 if not subsample else 2),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            BasicDeformConv2d(c_out, c_out, kernel_size=3,
                            stride=1),
            nn.BatchNorm2d(c_out)
        )

        # 1x1 convolution with stride 2 means we take the upper left value, and transform it to new output size
        self.downsample = nn.Conv2d(
            c_in, c_out, kernel_size=1, stride=2) if subsample else None
        self.act_fn = nn.ReLU(inplace=True)

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        out = self.act_fn(out)
        return out

class GatedDeformableResNetBlockEnc(nn.Module):

    def __init__(self, c_in, c_out, subsample=False):
        """
        Inputs:
            c_in - Number of input features
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in

        # Network representing F

        self.net = nn.Sequential(
            # No bias needed as the Batch Norm handles it
            GatedConv2d(c_in, c_out, kernel_size=3,
                            stride=1 if not subsample else 2),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            BasicDeformConv2d(c_out, c_out, kernel_size=3,
                            stride=1),
            nn.BatchNorm2d(c_out)
        )

        # 1x1 convolution with stride 2 means we take the upper left value, and transform it to new output size
        self.downsample = nn.Conv2d(
            c_in, c_out, kernel_size=1, stride=2) if subsample else None
        self.act_fn = nn.ReLU(inplace=True)

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        out = self.act_fn(out)
        return out