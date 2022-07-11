import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from sklearn.preprocessing import KBinsDiscretizer

from data import IqScaler, RegressionScaler
from perceiver_io import PerceiverEncoder, PerceiverDecoder, SASPerceiverIO, TaskDecoder


def multitask_l1(pred: torch.Tensor, target: torch.Tensor):
    valid_idx = torch.bitwise_not(torch.isnan(target))
    return F.l1_loss(pred[valid_idx], target[valid_idx], reduction='mean')


def multitask_mse(pred: torch.Tensor, target: torch.Tensor):
    valid_idx = torch.bitwise_not(torch.isnan(target))
    return F.mse_loss(pred[valid_idx], target[valid_idx])


class SASPerceiverIOModel(pl.LightningModule):

    def __init__(self,
                 num_classes: int,
                 num_reg_outputs: int,
                 latent_dim: int = 256,
                 enc_num_self_attn_per_block: int = 4,
                 enc_num_cross_attn_heads: int = 1,
                 enc_num_self_attn_heads: int = 2,
                 enc_cross_attn_widening_factor: int = 1,
                 enc_self_attn_widening_factor: int = 1,
                 enc_dropout: float = 0.1,
                 enc_cross_attention_dropout: float = 0.1,
                 enc_self_attention_dropout: float = 0.1,
                 model_dec_widening_factor: int = 1,
                 model_dec_num_heads: int = 1,
                 model_dec_qk_out_dim: int = 64,
                 model_dec_dropout: float = 0.1,
                 model_dec_attn_dropout: float = 0.1,
                 param_dec_widening_factor: int = 1,
                 param_dec_num_heads: int = 2,
                 param_dec_qk_out_dim: int = 256,
                 param_dec_dropout: float = 0.2,
                 param_dec_attn_dropout: float = 0.2,
                 lr: float = 5e-4,
                 batch_size: int = 256,
                 weight_decay: float = 1e-8,
                 n_bins: int = 640,
                 clf_weight: float = 1.0,
                 reg_weight: float = 1.0,
                 x_scaler: IqScaler = None,
                 y_scaler: RegressionScaler = None,
                 discretizer: KBinsDiscretizer = None):
        super().__init__()
        self.clf_weight = clf_weight
        self.reg_weight = reg_weight
        # scalers/preprocessors only for inference
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.discretizer = discretizer
        # metrics
        self.num_classes = num_classes
        self.save_hyperparameters(ignore=['model'])
        # encoder
        self.encoder = PerceiverEncoder(num_latents=latent_dim,
                                        latent_dim=latent_dim,
                                        input_dim=1,
                                        num_self_attn_per_block=enc_num_self_attn_per_block,
                                        num_cross_attn_heads=enc_num_cross_attn_heads,
                                        num_self_attn_heads=enc_num_self_attn_heads,
                                        cross_attn_widening_factor=enc_cross_attn_widening_factor,
                                        self_attn_widening_factor=enc_self_attn_widening_factor,
                                        dropout=enc_dropout,
                                        cross_attention_dropout=enc_cross_attention_dropout,
                                        self_attention_dropout=enc_self_attention_dropout)
        # clf decoder
        self.sas_model_decoder = TaskDecoder(num_outputs=num_classes,
                                             latent_dim=latent_dim,
                                             widening_factor=model_dec_widening_factor,
                                             num_heads=model_dec_num_heads,
                                             qk_out_dim=model_dec_qk_out_dim,
                                             dropout=model_dec_dropout,
                                             attention_dropout=model_dec_attn_dropout)
        # reg decoder
        self.sas_param_decoder = TaskDecoder(num_outputs=num_reg_outputs,
                                             latent_dim=latent_dim,
                                             widening_factor=param_dec_widening_factor,
                                             num_heads=param_dec_num_heads,
                                             qk_out_dim=param_dec_qk_out_dim,
                                             dropout=param_dec_dropout,
                                             attention_dropout=param_dec_attn_dropout)
        self.perceiver = SASPerceiverIO(
            self.encoder, self.sas_model_decoder, self.sas_param_decoder, n_bins)

    def forward(self, x):
        return self.perceiver(x)

    def training_step(self, batch, batch_idx):
        x, y_clf_true, y_reg_true = batch
        y_clf_pred, y_reg_pred = self(x)
        clf_loss = F.cross_entropy(y_clf_pred, y_clf_true)
        reg_loss = multitask_l1(y_reg_pred, y_reg_true)
        loss = self.clf_weight*clf_loss + self.reg_weight*reg_loss
        acc = accuracy(torch.argmax(y_clf_pred, dim=1),
                       y_clf_true, num_classes=self.num_classes)
        current_lr = self.trainer.optimizers[0].state_dict()[
            "param_groups"][0]["lr"]

        self.log_losses_and_metrics(
            clf_loss, reg_loss, acc, current_lr, mode='train')
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

    def test_step(self, batch, batch_idx):
        x, y_clf_true, y_reg_true = batch
        y_clf_pred, y_reg_pred = self(x)
        clf_loss = F.cross_entropy(y_clf_pred, y_clf_true)
        reg_loss = multitask_l1(y_reg_pred, y_reg_true)
        acc = accuracy(torch.argmax(y_clf_pred, dim=1),
                       y_clf_true, num_classes=self.num_classes)

        self.log_losses_and_metrics(clf_loss, reg_loss, acc, mode='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.lr,
                                     weight_decay=self.hparams.weight_decay)
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                     warmup_epochs=int(
                                                         0.05*self.trainer.max_epochs),
                                                     max_epochs=self.trainer.max_epochs,
                                                     eta_min=1e-7)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': lr_scheduler}}

    def log_losses_and_metrics(self, clf_loss, reg_loss, acc, lr=None, mode='train'):
        self.log(f'{mode}/clf_loss', self.clf_weight*clf_loss,
                 on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{mode}/reg_loss', self.reg_weight*reg_loss,
                 on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{mode}/accuracy', acc,
                 on_epoch=True, on_step=False, prog_bar=True)  # torchmetrics doesn't need sync_dist
        self.log(f'{mode}/mae', reg_loss,
                 on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        if lr is not None and mode == 'train':
            self.log('trainer/lr', lr, on_epoch=True,
                     on_step=False, rank_zero_only=True)
