# from modules import EdgeAttentionModule, ResNetBlockDec
import torch
import torch.nn as nn
from modules.blocks import ResNetBlockDec


class AttentionDecoder(nn.Module):
    def __init__(self, gated=False):
        super().__init__()
        self._create_network(gated)
        self._init_params()

    def  _create_network(self, gated):
        self.to_hidden = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.attn1 = EdgeAttentionModule(64, 32, gated)
        self.attn2 = EdgeAttentionModule(32, 16, gated)
        self.attn3 = EdgeAttentionModule(16, 16, gated)

        self.conv_feat = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.conv_edge = nn.Conv2d(16, 2, kernel_size=3, padding=1)
    

    def forward(self, feat):

        feat = self.to_hidden(feat)
        feat, edges = self.attn1(feat, feat)
        feat, edges = self.attn2(feat, edges)
        feat, edges = self.attn3(feat, edges)


        feat = self.conv_feat(feat)
        edges = self.conv_edge(edges)

        return feat, edges

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class EdgeAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, gated):
        super().__init__()

        self.dec_block = nn.Sequential(
            ResNetBlockDec(in_channels, in_channels, subsample=False, gated=gated),
            ResNetBlockDec(in_channels, out_channels, subsample=True, gated=gated)
        )

        self.edge_block = nn.Sequential(
            ResNetBlockDec(in_channels, in_channels, subsample=False, gated=gated),
            ResNetBlockDec(in_channels, out_channels, subsample=True, gated=gated)
        )
        

    def forward(self, feat, edges):

        feat = self.dec_block(feat)
        edges = self.edge_block(edges)

        attention = torch.sigmoid(edges)

        feat = attention * feat + feat

        return feat, edges