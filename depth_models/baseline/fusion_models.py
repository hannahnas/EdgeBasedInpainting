from decoder import ResNetDecoder
from modules.encoder import GatedResNetEncoder, ResNetEncoder, DeformableResNetEncoder
from skip_resnet import SkipResNet
import pytorch_lightning as pl
from modules.criterions import L1_loss
from modules.evaluation import compute_eval_measures
from torch import optim
import torch


class EarlyFusionModel(pl.LightningModule):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params
        if self.hyper_params['gated']:
            encoder = GatedResNetEncoder
        if self.hyper_params['deformable']:
            encoder = DeformableResNetEncoder()
        else:
            encoder = ResNetEncoder
        
        self.encoder = encoder(in_channels=4)
        self.decoder = ResNetDecoder(out_channels=1)


    def forward(self, batch):
        color = batch['rgb']
        depth = batch['depth']
        mask = batch['mask']
        depth_in = (1 - mask) * depth
        
        x = torch.cat((color, depth_in), dim=1)
        x = self.encoder(x)
        x = self.decoder(x)

        return x
        

    def _get_reconstruction_loss(self, batch):
        
        depth_gt = batch['depth']
        depth_pred = self.forward(batch)
        mask = batch['mask']

        l1_depth = L1_loss(depth_pred, depth_gt, mask)

        return l1_depth

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)

        return {"optimizer": optimizer, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        depth_loss = self._get_reconstruction_loss(batch)
        self.log('train_depth_loss', depth_loss)

        return depth_loss

    def validation_step(self, batch, batch_idx):
        depth_loss = self._get_reconstruction_loss(batch)
        self.log('val_depth_loss', depth_loss)


    def test_step(self, batch, batch_idx):
        depth_loss = self._get_reconstruction_loss(batch)
        self.log('test_depth_loss', depth_loss)


class MiddleFusionModel(pl.LightningModule):
    def __init__(self, hyper_params):
        super().__init__()
        self.hyper_params = hyper_params

        self.depth_encoder = ResNetEncoder(in_channels=1)
        self.rgb_encoder = ResNetEncoder(in_channels=3)

        # self.depth_encoder = ResNetEncoder(in_channels=1)
        self.decoder = ResNetDecoder(out_channels=1)


    def forward(self, batch):
        color = batch['rgb']
        depth = batch['depth']
        mask = batch['mask']
        depth_in = (1 - mask) * depth

        depth_feat = self.depth_encoder(depth_in)
        color_feat = self.rgb_encoder(color)
        
        x = torch.cat((color_feat, depth_feat), dim=1)

        x = self.decoder(x)

        return x
        

    def _get_reconstruction_loss(self, batch):
        
        depth_gt = batch['depth']
        depth_pred = self.forward(batch)
        mask = batch['mask']

        l1_depth = L1_loss(depth_pred, depth_gt, mask)

        return l1_depth

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hyper_params['lr'])

        return {"optimizer": optimizer, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        depth_loss = self._get_reconstruction_loss(batch)
        self.log('train_depth_loss', depth_loss)

        return depth_loss

    def validation_step(self, batch, batch_idx):
        depth_loss = self._get_reconstruction_loss(batch)
        self.log('val_depth_loss', depth_loss)


    def test_step(self, batch, batch_idx):
        depth_loss = self._get_reconstruction_loss(batch)
        self.log('test_depth_loss', depth_loss)
        depth_gt = batch['depth']
        mask = batch['mask']

        with torch.no_grad():
            depth_pred = self.forward(batch)
        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_eval_measures(depth_gt, depth_pred, mask)

        self.log('abs rel', abs_rel)
        self.log('sq rel', sq_rel)
        self.log('rmse', rmse)
        self.log('rmse log', rmse_log)
        self.log('delta 1.25', a1)
        self.log('delta 1.25^2', a2)
        self.log('delta 1.25^3', a3)



class SkipResNetModel(pl.LightningModule):
    def __init__(self, deformable, use_edges):
        super().__init__()
        
        self.skipnet = SkipResNet(deformable, use_edges)


    def forward(self, batch):

        return self.skipnet(batch)
        

    def _get_reconstruction_loss(self, batch):
        
        depth_gt = batch['depth']
        depth_pred = self.forward(batch)
        mask = batch['mask']

        l1_depth = L1_loss(depth_pred, depth_gt, mask)

        return l1_depth

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)

        return {"optimizer": optimizer, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        depth_loss = self._get_reconstruction_loss(batch)
        self.log('train_depth_loss', depth_loss)

        return depth_loss

    def validation_step(self, batch, batch_idx):
        depth_loss = self._get_reconstruction_loss(batch)
        self.log('val_depth_loss', depth_loss)


    def test_step(self, batch, batch_idx):
        depth_loss = self._get_reconstruction_loss(batch)
        self.log('test_depth_loss', depth_loss)

        depth_gt = batch['depth']
        mask = batch['mask']
        with torch.no_grad():
            depth_pred = self.forward(batch)

        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_eval_measures(depth_gt, depth_pred, mask)

        self.log('abs rel', abs_rel)
        self.log('sq rel', sq_rel)
        self.log('rmse', rmse)
        self.log('rmse log', rmse_log)
        self.log('delta 1.25', a1)
        self.log('delta 1.25^2', a2)
        self.log('delta 1.25^3', a3)
        

