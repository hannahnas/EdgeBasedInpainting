import torch
import torch.nn as nn
from modules.blocks import ResNetBlockDec
from depth_models.skip_attention.skipattention import SkipEncoder

class SkipNet(nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.rgb_encoder = SkipEncoder(in_channels=3, type=hyper_params['encoder type'])
        
        if hyper_params['use edges']:
            self.depth_encoder = SkipEncoder(in_channels=2, type=hyper_params['encoder type'])
        else:
            self.depth_encoder = SkipEncoder(in_channels=1, type=hyper_params['encoder type'])

        self._create_network(out_channels=1)
        self._init_params()

    def _create_network(self, out_channels):

        self.reduce_features = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.dec_layer_3 = nn.Sequential(
            ResNetBlockDec(256, 256, subsample=False),
            ResNetBlockDec(256, 128, subsample=True)
        )

        self.dec_layer_2 = nn.Sequential(
            ResNetBlockDec(128, 128, subsample=False),
            ResNetBlockDec(128, 64, subsample=True)
        )

        self.dec_layer_1 = nn.Sequential(
            ResNetBlockDec(64, 64, subsample=False),
            ResNetBlockDec(64, 32, subsample=True)
        )

        self.dec_layer_0 = nn.Conv2d(32, out_channels, 3, padding=1)


    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, rgb, depth):

        rgb_x1, rgb_x2, rgb_x3 = self.rgb_encoder(rgb)
        depth_x1, depth_x2, depth_x3 = self.depth_encoder(depth)

        features = torch.cat([rgb_x3, depth_x3], dim=1)
        features = self.reduce_features(features)

        dec_x2 = self.dec_layer_3(features)
        
        # skip connection
        skip_dec_x2 = dec_x2 + rgb_x2 + depth_x2 # shape: B, 124, 52, 52
        dec_x1 = self.dec_layer_2(skip_dec_x2)

        # skip connection
        skip_dec_x1 = dec_x1 + rgb_x1 + depth_x1 # shape: B, 62, 104, 104

        dec_x0 = self.dec_layer_1(skip_dec_x1)

        depth_pred = self.dec_layer_0(dec_x0)

        return depth_pred
        