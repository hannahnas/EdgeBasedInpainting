import torch
import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as F
from modules.encoder import ResNetEncoder
from depth_models.baseline.decoder import ResNetDecoder
from modules.evaluation import compute_eval_measures


class BaselineModel(pl.LightningModule):
    def __init__(self, hyper_params):
        super().__init__()

        self.hyper_params = hyper_params
        encoder_type = hyper_params['encoder type']

        self.rgb_encoder = ResNetEncoder(in_channels=3, type=encoder_type)
        self.depth_encoder = ResNetEncoder(in_channels=1, type=encoder_type)

        self.decoder = ResNetDecoder(in_channels=512, out_channels=1)
        

    def forward(self, batch):
        rgb = batch['rgb']
        masked_depth = (1 - batch['mask']) * batch['depth']

        rgb_feat = self.rgb_encoder(rgb)
        depth_feat = self.depth_encoder(masked_depth)

        feat = torch.cat([rgb_feat, depth_feat], dim=1)
        pred = self.decoder(feat)

        return pred
    
    def _get_loss(self, batch):
        depth_gt = batch['depth']
        mask = batch['mask']

        depth_pred = self.forward(batch)
        l1_loss = F.l1_loss(depth_pred * mask, depth_gt * mask, reduction='mean') / mask.sum()

        self.log('train_l1_loss', l1_loss)
        return l1_loss
   
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hyper_params['lr'])

        return {"optimizer": optimizer, "monitor": "total_val_loss"}

    def training_step(self, batch, batch_idx):
        l1_loss = self._get_loss(batch)
        self.log('train_l1_depth_loss', l1_loss)

        return l1_loss

    def validation_step(self, batch, batch_idx):
        l1_loss = self._get_loss(batch)
        self.log('val_l1_depth_loss', l1_loss)


    def test_step(self, batch, batch_idx):
        l1_loss = self._get_loss(batch)
        self.log('test_l1_depth_loss', l1_loss)

        depth_pred = self.forward(batch)
        depth_gt = batch['depth']
        mask = batch['mask']
        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_eval_measures(depth_gt, depth_pred, mask)

        self.log('abs rel', abs_rel)
        self.log('sq rel', sq_rel)
        self.log('rmse', rmse)
        self.log('rmse log', rmse_log)
        self.log('delta 1.25', a1)
        self.log('delta 1.25^2', a2)
        self.log('delta 1.25^3', a3)