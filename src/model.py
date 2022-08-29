import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from sklearn.preprocessing import KBinsDiscretizer

from data import IqTransformer, TargetTransformer
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
                 num_latents: int = 64,
                 enc_num_blocks: int = 4,
                 enc_num_self_attn_per_block: int = 6,
                 enc_num_cross_attn_heads: int = 2,
                 enc_num_self_attn_heads: int = 4,
                 enc_cross_attn_widening_factor: int = 1,
                 enc_self_attn_widening_factor: int = 3,
                 enc_dropout: float = 0.1,
                 enc_cross_attention_dropout: float = 0.1,
                 enc_self_attention_dropout: float = 0.1,
                 model_dec_widening_factor: int = 3,
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
                 weight_decay: float = 0.0,
                 n_bins: int = 256,
                 clf_weight: float = 1.0,
                 reg_weight: float = 1.0,
                 input_transformer: IqTransformer = None,
                 target_transformer: TargetTransformer = None):
        super().__init__()
        self.clf_weight = clf_weight
        self.reg_weight = reg_weight
        # scalers/preprocessors only for inference
        self.input_transformer = input_transformer
        self.target_transformer = target_transformer
        self.log_indices_ = None
        self.scaler_mu_ = None
        self.scaler_std_ = None

        # metrics
        self.num_classes = num_classes
        self.save_hyperparameters(ignore=['model'])
        # encoder
        self.encoder = PerceiverEncoder(num_latents=num_latents,
                                        latent_dim=latent_dim,
                                        num_blocks=enc_num_blocks,
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
        mae = multitask_l1(self.unscale_y(y_reg_pred),
                           self.unscale_y(y_reg_true))
        current_lr = self.trainer.optimizers[0].state_dict()[
            "param_groups"][0]["lr"]

        # self.log_losses_and_metrics(
        #     clf_loss, reg_loss, acc, mae, current_lr, mode='train')
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_clf_true, y_reg_true = batch
        y_clf_pred, y_reg_pred = self(x)
        clf_loss = F.cross_entropy(y_clf_pred, y_clf_true)
        reg_loss = multitask_l1(y_reg_pred, y_reg_true)
        loss = self.clf_weight*clf_loss + self.reg_weight*reg_loss
        acc = accuracy(torch.argmax(y_clf_pred, dim=1),
                       y_clf_true, num_classes=self.num_classes)
        mae = multitask_l1(self.unscale_y(y_reg_pred),
                           self.unscale_y(y_reg_true))

        # self.log_losses_and_metrics(clf_loss, reg_loss, acc, mae, mode='val')

    def test_step(self, batch, batch_idx):
        x, y_clf_true, y_reg_true = batch
        y_clf_pred, y_reg_pred = self(x)
        clf_loss = F.cross_entropy(y_clf_pred, y_clf_true)
        reg_loss = multitask_l1(y_reg_pred, y_reg_true)
        acc = accuracy(torch.argmax(y_clf_pred, dim=1),
                       y_clf_true, num_classes=self.num_classes)
        mae = multitask_l1(self.unscale_y(y_reg_pred),
                           self.unscale_y(y_reg_true))

        # self.log_losses_and_metrics(clf_loss, reg_loss, acc, mae, mode='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.lr,
                                     weight_decay=self.hparams.weight_decay,
                                     #  eps=1e-4  # for half-precision
                                     )
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                     warmup_epochs=int(
                                                         0.05*self.trainer.max_epochs),
                                                     max_epochs=self.trainer.max_epochs,
                                                     eta_min=1e-6)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': lr_scheduler}}

    def log_losses_and_metrics(self, clf_loss, reg_loss, acc, mae, lr=None, mode='train'):
        total_loss = self.reg_weight*reg_loss+self.clf_weight*clf_loss
        self.log(f'{mode}/total_loss', total_loss,
                 on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{mode}/clf_loss', self.clf_weight*clf_loss,
                 on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{mode}/reg_loss', self.reg_weight*reg_loss,
                 on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{mode}/accuracy', acc,
                 on_epoch=True, on_step=False, prog_bar=True)  # torchmetrics doesn't need sync_dist
        self.log(f'{mode}/mae', mae,
                 on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        if lr is not None and mode == 'train':
            self.log('trainer/lr', lr, on_epoch=True,
                     on_step=False, rank_zero_only=True)

    def unscale_y(self, y):
        if self.log_indices_ is None:
            self.log_indices_ = torch.Tensor(
                self.target_transformer.log_indices).type_as(y).long()
            self.scaler_mu_ = torch.Tensor(
                self.target_transformer.scaler.mean_).type_as(y)
            self.scaler_std_ = torch.Tensor(
                self.target_transformer.scaler.scale_).type_as(y)

        y = y*self.scaler_std_ + self.scaler_mu_
        # y = y.float()  # needed when training with half-precision
        y[:, self.log_indices_] = torch.exp(y[:, self.log_indices_])
        return y
