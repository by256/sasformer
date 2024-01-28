import pytorch_lightning as pl
import torch
import torch.nn.functional as F  # noqa: N812
from perceiver_io.encoder import PerceiverEncoder
from perceiver_io.decoders import PerceiverDecoder
from perceiver_io.perceiver import SASPerceiverIO
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics.functional import accuracy, mean_absolute_percentage_error

from data import IqTransformer, TargetTransformer


def multitask_l1(pred: torch.Tensor, target: torch.Tensor) -> float:
    valid_idx = torch.bitwise_not(torch.isnan(target))
    return F.l1_loss(pred[valid_idx], target[valid_idx], reduction="mean")


def multitask_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    valid_idx = torch.bitwise_not(torch.isnan(target))
    return F.mse_loss(pred[valid_idx], target[valid_idx])


def multitask_mape(pred: torch.Tensor, target: torch.Tensor) -> float:
    valid_idx = torch.bitwise_not(torch.isnan(target))
    return mean_absolute_percentage_error(pred[valid_idx], target[valid_idx])


class SASPerceiverIOModel(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        num_reg_outputs: int,
        seq_len=511,
        latent_dim: int = 256,
        num_latents: int = 64,
        enc_num_blocks: int = 1,
        enc_num_self_attn_per_block: int = 6,
        enc_num_cross_attn_heads: int = 2,
        enc_num_self_attn_heads: int = 4,
        enc_cross_attn_widening_factor: int = 1,
        enc_self_attn_widening_factor: int = 3,
        enc_dropout: float = 0.1,
        enc_cross_attn_dropout: float = 0.1,
        enc_self_attn_dropout: float = 0.1,
        model_dec_widening_factor: int = 3,
        model_dec_num_heads: int = 5,
        model_dec_dropout: float = 0.1,
        model_dec_attn_dropout: float = 0.1,
        param_dec_widening_factor: int = 1,
        param_dec_num_heads: int = 3,
        param_dec_dropout: float = 0.2,
        param_dec_attn_dropout: float = 0.2,
        lr: float = 5e-4,
        batch_size: int = 256,
        weight_decay: float = 1e-7,
        n_bins: int = 256,
        clf_weight: float = 1.0,
        reg_weight: float = 1.0,
        reg_obj: str = "mae",
        input_transformer: IqTransformer = None,
        target_transformer: TargetTransformer = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.clf_weight = clf_weight
        self.reg_weight = reg_weight
        self.reg_obj = reg_obj
        # scalers/preprocessors only for inference
        self.input_transformer = input_transformer
        self.target_transformer = target_transformer
        self._target_transformer_buffers_registered = False
        # metrics
        self.save_hyperparameters(ignore=["model"])
        # encoder
        self.encoder = PerceiverEncoder(
            num_latents=num_latents,
            latent_dim=latent_dim,
            num_blocks=enc_num_blocks,
            num_self_attn_per_block=enc_num_self_attn_per_block,
            num_cross_attn_heads=enc_num_cross_attn_heads,
            num_self_attn_heads=enc_num_self_attn_heads,
            cross_attn_widening_factor=enc_cross_attn_widening_factor,
            self_attn_widening_factor=enc_self_attn_widening_factor,
            dropout=enc_dropout,
            cross_attn_dropout=enc_cross_attn_dropout,
            self_attn_dropout=enc_self_attn_dropout,
        )
        # clf decoder
        self.sas_model_decoder = PerceiverDecoder(
            num_outputs=num_classes,
            num_latents=1,
            latent_dim=latent_dim,
            widening_factor=model_dec_widening_factor,
            num_heads=model_dec_num_heads,
            dropout=model_dec_dropout,
            attn_dropout=model_dec_attn_dropout,
        )
        # reg decoder
        self.sas_param_decoder = PerceiverDecoder(
            num_outputs=num_reg_outputs,
            num_latents=1,
            latent_dim=latent_dim,
            widening_factor=param_dec_widening_factor,
            num_heads=param_dec_num_heads,
            dropout=param_dec_dropout,
            attn_dropout=param_dec_attn_dropout,
        )

        self.perceiver = SASPerceiverIO(self.encoder, self.sas_model_decoder, self.sas_param_decoder, n_bins, seq_len)

    def forward(self, x, mask=None):
        return self.perceiver(x, mask)

    def training_step(self, batch, batch_idx):
        return self.step_(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.step_(batch, batch_idx, mode="val")

    def test_step(self, batch, batch_idx):
        return self.step_(batch, batch_idx, mode="test")

    def predict_step(self, batch, batch_idx):
        return self.step_(batch, batch_idx, mode="predict")

    def step_(self, batch, batch_idx, mode):
        x, y_clf_true, y_reg_true = batch[:3]
        mask = batch[3] if len(batch) == 4 else None
        # forward
        y_clf_pred, y_reg_pred = self(x, mask)
        # calculate individual losses
        clf_loss = F.cross_entropy(y_clf_pred, y_clf_true)
        if self.reg_obj == "mse":
            reg_loss = multitask_mse(y_reg_pred, y_reg_true)
        elif self.reg_obj == "mae":
            reg_loss = multitask_l1(y_reg_pred, y_reg_true)
        # calc and log metrics and losses
        with torch.no_grad():
            acc = accuracy(
                torch.argmax(y_clf_pred, dim=1),
                y_clf_true,
                num_classes=self.num_classes,
                task="multiclass",
            )
            mape = multitask_mape(y_reg_pred, y_reg_true)
            # mae = multitask_l1(self.unscale_y(y_reg_pred), self.unscale_y(y_reg_true))
            if mode == "train":
                current_lr = self.trainer.optimizers[0].state_dict()["param_groups"][0]["lr"]
            else:
                current_lr = None
        if not mode == "predict":
            self.log_losses_and_metrics(clf_loss, reg_loss, acc, mape, current_lr, mode=mode)

        if mode in ["train", "val"]:
            loss = self.clf_weight * clf_loss + self.reg_weight * reg_loss
            return loss
        elif mode == "predict":
            return y_clf_pred, y_reg_pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=int(0.05 * self.trainer.max_epochs),
            max_epochs=self.trainer.max_epochs,
            eta_min=self.hparams.lr / 100.0,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler}}

    def log_losses_and_metrics(self, clf_loss, reg_loss, acc, mape, lr=None, mode="train"):
        total_loss = self.reg_weight * reg_loss + self.clf_weight * clf_loss
        self.log(f"{mode}/total_loss", total_loss, on_epoch=True, on_step=False, rank_zero_only=True)
        self.log(
            f"{mode}/clf_loss",
            self.clf_weight * clf_loss,
            on_epoch=True,
            on_step=False,
            rank_zero_only=True,
        )
        self.log(
            f"{mode}/reg_loss",
            self.reg_weight * reg_loss,
            on_epoch=True,
            on_step=False,
            rank_zero_only=True,
        )
        self.log(
            f"{mode}/accuracy",
            acc,
            on_epoch=True,
            on_step=False,
            rank_zero_only=True,
            prog_bar=True,
        )
        self.log(f"{mode}/mape", mape, on_epoch=True, on_step=False, prog_bar=True, rank_zero_only=True)
        # self.log(f"{mode}/mae", mae, on_epoch=True, on_step=False, prog_bar=True, rank_zero_only=True)
        if lr is not None and mode == "train":
            self.log("trainer/lr", lr, on_epoch=True, on_step=False, rank_zero_only=True)

    def unscale_y(self, y):
        if not self._target_transformer_buffers_registered:
            log_indices_ = torch.Tensor(self.target_transformer.log_indices).type_as(y).long()
            self.register_buffer("log_indices_", log_indices_, persistent=False)
            scaler_mu_ = torch.Tensor(self.target_transformer.scaler.mean_).type_as(y)
            self.register_buffer("scaler_mu_", scaler_mu_, persistent=False)
            scaler_std_ = torch.Tensor(self.target_transformer.scaler.scale_).type_as(y)
            self.register_buffer("scaler_std_", scaler_std_, persistent=False)
            self._target_transformer_buffers_registered = True

        y = y * self.scaler_std_ + self.scaler_mu_
        y[:, self.log_indices_] = torch.exp(y[:, self.log_indices_])
        return y
