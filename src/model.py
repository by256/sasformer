import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from data import IqScaler, RegressionScaler


def multitask_l1(pred: torch.Tensor, target: torch.Tensor):
    valid_idx = torch.bitwise_not(torch.isnan(target))
    return F.l1_loss(pred[valid_idx], target[valid_idx], reduction='mean')


def multitask_mse(pred: torch.Tensor, target: torch.Tensor):
    valid_idx = torch.bitwise_not(torch.isnan(target))
    return F.mse_loss(pred[valid_idx], target[valid_idx])


class LightningModel(pl.LightningModule):

    def __init__(self,
                 model: nn.Module,
                 num_classes: int,
                 lr: float = 5e-4,
                 weight_decay: float = 1e-7,
                 clf_weight: float = 1.0,
                 reg_weight: float = 1.0,
                 x_scaler: IqScaler = None,
                 y_scaler: RegressionScaler = None):
        super().__init__()
        self.model = model
        self.clf_weight = clf_weight
        self.reg_weight = reg_weight
        # scalers only for inference
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        # metrics
        self.num_classes = num_classes
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y_clf_true, y_reg_true = batch
        y_clf_pred, y_reg_pred = self(x)
        clf_loss = F.cross_entropy(y_clf_pred, y_clf_true)
        reg_loss = multitask_l1(y_reg_pred, y_reg_true)
        loss = self.clf_weight*clf_loss + self.reg_weight*reg_loss
        acc = accuracy(torch.argmax(y_clf_pred, dim=1),
                       y_clf_true, num_classes=self.num_classes)

        self.log_losses_and_metrics(clf_loss, reg_loss, acc, mode='train')
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_clf_true, y_reg_true = batch
        y_clf_pred, y_reg_pred = self(x)
        clf_loss = F.cross_entropy(y_clf_pred, y_clf_true)
        reg_loss = multitask_l1(y_reg_pred, y_reg_true)
        loss = self.clf_weight*clf_loss + self.reg_weight*reg_loss
        acc = accuracy(torch.argmax(y_clf_pred, dim=1),
                       y_clf_true, num_classes=self.num_classes)

        self.log_losses_and_metrics(clf_loss, reg_loss, acc, mode='val')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.lr,
                                     weight_decay=self.hparams.weight_decay)
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                     warmup_epochs=10,
                                                     max_epochs=self.trainer.max_epochs)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': lr_scheduler}}

    def log_losses_and_metrics(self, clf_loss, reg_loss, acc, mode='train'):
        if 'ddp' in self.trainer.strategy.strategy_name:
            self.log(f'{mode}_clf_loss', self.clf_weight*clf_loss,
                     on_step=False, on_epoch=True, sync_dist=True)
            self.log(f'{mode}_reg_loss', self.reg_weight*reg_loss,
                     on_step=False, on_epoch=True, sync_dist=True)
            self.log(f'{mode}_accuracy', acc, on_step=False,
                     on_epoch=True, sync_dist=True)
            self.log(f'{mode}_mae', reg_loss, on_step=False,
                     on_epoch=True, sync_dist=True)
        else:
            self.logger.experiment.add_scalars(
                'clf_loss', {mode: self.clf_weight*clf_loss}, global_step=self.current_epoch)
            self.logger.experiment.add_scalars(
                'reg_loss', {mode: self.reg_weight*reg_loss}, global_step=self.current_epoch)
            self.logger.experiment.add_scalars(
                'accuracy', {mode: acc}, global_step=self.current_epoch)
            self.logger.experiment.add_scalars(
                'mae', {mode: reg_loss}, global_step=self.current_epoch)
