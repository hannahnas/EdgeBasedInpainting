import torch
import torch.nn as nn
from modules.dataset import InpaintDataset
from modules.blocks import DeformableResNetBlockEnc, GatedResNetBlockEnc, ResNetBlockEnc, ResNetBlockDec, GatedDeformableResNetBlockEnc
from modules.convs import GatedConv2d, BasicDeformConv2d
from torch.utils.data import DataLoader


class SkipEncoder(nn.Module):
    def __init__(self, in_channels, type='base'):
        super().__init__()
        self._create_network(in_channels, type=type)
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
        x0 = self.layer_0(x)
        x1 = self.layer_1(x0)
        x2 = self.layer_2(x1)
        x3 = self.layer_3(x2)

        return x1, x2, x3



class SkipAttentionNet(nn.Module):
    def __init__(self, hyper_params):
        super().__init__()
        self.edge_encoder = SkipEncoder(in_channels=1)
        self.rgb_encoder = SkipEncoder(in_channels=3, type=hyper_params['encoder type'])
        self.depth_encoder = SkipEncoder(in_channels=1, type=hyper_params['encoder type'])

        self.multiscale = hyper_params['multiscale']

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

        if self.multiscale:
            self.scale_64 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
            self.scale_128 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, edges, rgb, depth):

        edge_x1, edge_x2, edge_x3 = self.edge_encoder(edges)
        rgb_x1, rgb_x2, rgb_x3 = self.rgb_encoder(rgb)
        depth_x1, depth_x2, depth_x3 = self.depth_encoder(depth)

        features = torch.cat([rgb_x3, depth_x3], dim=1)
        features = self.reduce_features(features)

        dec_x2 = self.dec_layer_3(features)
        
        # skip attention
        attn2 = torch.sigmoid(edge_x2)
        attn_dec_x2 = dec_x2 + (attn2 * rgb_x2) + (attn2 * depth_x2) # shape: B, 128, 52, 52

        dec_x1 = self.dec_layer_2(attn_dec_x2)
        # skip attention
        attn1 = torch.sigmoid(edge_x1)
        attn_dec_x1 = dec_x1 + (attn1 * rgb_x1) + (attn1 * depth_x1) # shape: B, 64, 104, 104

        dec_x0 = self.dec_layer_1(attn_dec_x1)

        depth_pred = self.dec_layer_0(dec_x0)

        if self.multiscale:
            out_64 = self.scale_64(attn_dec_x2)
            out_128 = self.scale_128(attn_dec_x1)
            out_256 = depth_pred

            return out_64, out_128, out_256

        return depth_pred
        

if __name__ == '__main__':
    train_set = InpaintDataset(split = 'train', samples=4)
    loader = DataLoader(train_set, batch_size=4, shuffle=True,
                            drop_last=True, pin_memory=True, num_workers=1)

    model = SkipAttentionNet()

    for batch in loader:
        edges = batch['edges']
        rgb = batch['rgb']
        depth = batch['depth']
        mask = batch['mask']
        masked_depth = (1 - mask) * depth

        out = model(edges, rgb, masked_depth)
        print(out.shape)
        break

