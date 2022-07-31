import torch
import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as F
from depth_models.skip_attention.skipattention import SkipAttentionNet
from modules.evaluation import compute_eval_measures


class SkipAttentionModel(pl.LightningModule):
    def __init__(self, hyper_params):
        super().__init__()
        self.multiscale = hyper_params['multiscale']
        self.hyper_params = hyper_params
        self.model = SkipAttentionNet(hyper_params)
        

    def forward(self, batch):
        rgb = batch['rgb']
        depth = batch['depth']
        mask = batch['mask']
        edges = batch['edges']
        masked_depth = (1 - mask) * depth 

        depth = self.model(edges, rgb, masked_depth)

        return depth
    
    def _get_loss(self, batch):
        depth_gt = batch['depth']
        mask = batch['mask']

        depth_pred = self.forward(batch)
        if self.multiscale:
            _, _, _, size = depth_gt.shape
            out_64, out_128, out_256 = depth_pred
            gt_64 = F.interpolate(depth_gt, size=(size//4, size//4), mode='bilinear')
            mask_64 = F.interpolate(depth_gt, size=(size//4, size//4), mode='nearest')
            

            gt_128 = F.interpolate(depth_gt, size=(size//2, size//2), mode='bilinear')
            mask_128 = F.interpolate(depth_gt, size=(size//2, size//2), mode='nearest')

            size_64_loss = F.l1_loss(out_64 * mask_64, gt_64 * mask_64, reduction='mean') / mask_64.sum()
            self.log('l1_64', size_64_loss)
            size_128_loss = F.l1_loss(out_128 * mask_128, gt_128 * mask_128, reduction='mean') / mask_128.sum()
            self.log('l1_128', size_128_loss)
            size_256_loss = F.l1_loss(out_256 * mask, depth_gt * mask, reduction='mean') / mask.sum()
            self.log('l1_256', size_256_loss)
            
            l1_loss = size_64_loss + size_128_loss + size_256_loss

            return l1_loss

        l1_loss = F.l1_loss(depth_pred * mask, depth_gt * mask, reduction='mean') / mask.sum()
        # self.log('train_l1_loss', l1_loss)
        return l1_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hyper_params['lr'])

        return {"optimizer": optimizer, "monitor": "val_l1_depth_loss"}

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
        if self.multiscale:
            _, _, depth_pred = depth_pred
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