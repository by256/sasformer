import os
import sys
import yaml
import wandb
import argparse
import numpy as np
import pandas as pd
from typing import Union
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from model import SASPerceiverIOModel
from data import SASDataModule


def estimate_batch_size(model, datamodule):
    input_size = sys.getsizeof(
        datamodule.train_dataset[0][0].shape[0]) * 1e-6
    model_size = model.model_size  # Mb
    gpu_mem = torch.cuda.get_device_properties(0).total_memory * 1e-6
    batch_size_exponent = np.floor(
        np.log2(gpu_mem / (input_size + model_size))) - 1.0
    return int(2**batch_size_exponent)


def load_hparams_from_yaml(path):
    keys_to_remove = ['num_classes',
                      'num_reg_outputs',
                      'wandb_version',
                      '_wandb',
                      'x_scaler',
                      'y_scaler',
                      'discretizer']
    with open(path, 'r') as stream:
        params = yaml.safe_load(stream)
    # remove unwanted keys
    for key in keys_to_remove:
        if key in params:
            del params[key]
    for key in params.keys():
        params[key] = params[key]['value']
    return params


def load_hparams_from_namespace(namespace):
    hparams = {'num_latents': namespace.num_latents,
               'latent_dim': namespace.latent_dim,
               'enc_num_blocks': namespace.enc_num_blocks,
               'enc_num_self_attn_per_block': namespace.enc_num_self_attn_per_block,
               'enc_num_cross_attn_heads': namespace.enc_num_cross_attn_heads,
               'enc_num_self_attn_heads': namespace.enc_num_self_attn_heads,
               'enc_cross_attn_widening_factor': namespace.enc_cross_attn_widening_factor,
               'enc_self_attn_widening_factor': namespace.enc_self_attn_widening_factor,
               'enc_dropout': namespace.enc_dropout,
               'enc_cross_attention_dropout': namespace.enc_cross_attention_dropout,
               'enc_self_attention_dropout': namespace.enc_self_attention_dropout,
               'model_dec_widening_factor': namespace.model_dec_widening_factor,
               'model_dec_num_heads': namespace.model_dec_num_heads,
               'model_dec_qk_out_dim': namespace.model_dec_qk_out_dim,
               'model_dec_dropout': namespace.model_dec_dropout,
               'model_dec_attn_dropout': namespace.model_dec_attn_dropout,
               'param_dec_widening_factor': namespace.param_dec_widening_factor,
               'param_dec_num_heads': namespace.param_dec_num_heads,
               'param_dec_qk_out_dim': namespace.param_dec_qk_out_dim,
               'param_dec_dropout': namespace.param_dec_dropout,
               'param_dec_attn_dropout': namespace.param_dec_attn_dropout,
               'lr': namespace.lr,
               'batch_size': batch_size,
               'weight_decay': namespace.weight_decay,
               'n_bins': namespace.n_bins,
               'clf_weight': namespace.clf_weight,
               'reg_weight': namespace.reg_weight}
    return hparams


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/', type=str,
                        help='Directory containing data files.', metavar='data_dir')
    parser.add_argument('--sub_dir', default='large', type=str,
                        help='Directory containing parquet files within data_dir.', metavar='sub_dir')
    parser.add_argument('--project_name', default='sas-perceiver', type=str,
                        help='Project name for logging.', metavar='project_name')
    parser.add_argument('--val_size', default=0.25, type=float,
                        help='Proportion of data to split for validation', metavar='val_size')
    parser.add_argument('--log_dir', default='../logs/', type=str,
                        help='Logging directory for Tensorboard', metavar='log_dir')
    parser.add_argument('--from_yaml', default=None, type=str,
                        help='path to hparams yaml file.', metavar='from_yaml')
    parser.add_argument('--disable_logger', default=0, type=int,
                        help='disable logger for debugging.', metavar='disable_logger')
    # encoder args
    parser.add_argument('--num_latents', default=64,
                        type=int, metavar='num_latents')
    parser.add_argument('--latent_dim', default=256,
                        type=int, metavar='latent_dim')
    parser.add_argument('--enc_num_blocks', default=4,
                        type=int, metavar='enc_num_blocks')
    parser.add_argument('--enc_num_self_attn_per_block', default=5,
                        type=int, metavar='encoder_num_self_attn_per_block')
    parser.add_argument('--enc_num_self_attn_heads', default=1,
                        type=int, metavar='encoder_num_self_attn_heads')
    parser.add_argument('--enc_num_cross_attn_heads', default=2,
                        type=int, metavar='enc_num_cross_attn_heads')
    parser.add_argument('--enc_cross_attn_widening_factor', default=1,
                        type=int, metavar='enc_cross_attn_widening_factor')
    parser.add_argument('--enc_self_attn_widening_factor', default=2,
                        type=int, metavar='enc_self_attn_widening_factor')
    parser.add_argument('--enc_dropout', default=0.2,
                        type=float, metavar='enc_dropout')
    parser.add_argument('--enc_cross_attention_dropout', default=0.2,
                        type=float, metavar='enc_cross_attention_dropout')
    parser.add_argument('--enc_self_attention_dropout', default=0.2,
                        type=float, metavar='enc_self_attention_dropout')
    # model (clf) decoder args
    parser.add_argument('--model_dec_widening_factor', default=2,
                        type=int, metavar='model_dec_widening_factor')
    parser.add_argument('--model_dec_num_heads', default=1,
                        type=int, metavar='model_decoder_num_heads')
    parser.add_argument('--model_dec_qk_out_dim', default=64,
                        type=int, metavar='model_dec_qk_out_dim')
    parser.add_argument('--model_dec_dropout', default=0.2,
                        type=float, metavar='model_dec_dropout')
    parser.add_argument('--model_dec_attn_dropout', default=0.2,
                        type=float, metavar='model_dec_attn_dropout')
    # param (reg) decoder args
    parser.add_argument('--param_dec_widening_factor', default=1,
                        type=int, metavar='param_dec_widening_factor')
    parser.add_argument('--param_dec_num_heads', default=1,
                        type=int, metavar='param_decoder_num_heads')
    parser.add_argument('--param_dec_qk_out_dim', default=256,
                        type=int, metavar='param_dec_qk_out_dim')
    parser.add_argument('--param_dec_dropout', default=0.2,
                        type=float, metavar='param_dec_dropout')
    parser.add_argument('--param_dec_attn_dropout', default=0.2,
                        type=float, metavar='param_dec_attn_dropout')
    # datamodule args
    parser.add_argument('--subsample', default=None,
                        type=int, help='Subsample data (for debugging)', metavar='subsample')
    # lightning model args
    parser.add_argument('--n_bins', default=128,
                        type=int, help='n bins for input discretization.', metavar='n_bins')
    parser.add_argument('--clf_weight', default=1.0,
                        type=float, metavar='clf_weight')
    parser.add_argument('--reg_weight', default=1.0,
                        type=float, metavar='reg_weight')
    # lightning trainer args
    parser.add_argument('--batch_size', default=1024,
                        type=int, metavar='batch_size')
    parser.add_argument('--batch_size_auto', default=0,
                        type=int, metavar='batch_size_auto')
    parser.add_argument('--lr', default=2e-3, type=float, metavar='lr')
    parser.add_argument('--weight_decay', default=1e-8,
                        type=float, metavar='weight_decay')
    parser.add_argument('--max_epochs', default=200,
                        type=int, metavar='max_epochs')
    parser.add_argument('--gradient_clip_val', default=1.0,
                        type=float, metavar='gradient_clip_val')
    parser.add_argument('--ckpt_path', default=None, type=str,
                        help='Checkpoint path to resume training', metavar='ckpt_path')
    parser.add_argument('--gpus', default=1, type=int, metavar='gpus')
    parser.add_argument('--accumulate_grad_batches', default=None,
                        type=int, metavar='accumulate_grad_batches')
    parser.add_argument('--overfit_batches', default=0,
                        type=int, metavar='overfit_batches')
    parser.add_argument('--detect_anomaly', default=1,
                        type=int, metavar='detect_anomaly')
    parser.add_argument('--deterministic', default=1,
                        type=int, metavar='deterministic')
    parser.add_argument('--strategy', default=None, type=str,
                        help='Set to `ddp` for cluster training', metavar='strategy')
    parser.add_argument('--num_nodes', default=1, type=int,
                        help='N nodes for distributed training.', metavar='num_nodes')
    parser.add_argument('--seed', default=None, type=int,
                        help='Random seed.', metavar='seed')
    namespace = parser.parse_args()

    if namespace.seed is not None:
        pl.seed_everything(namespace.seed, workers=True)

    # define paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, namespace.data_dir)

    # place holder if batch_size_auto == True
    batch_size = 2 if namespace.batch_size_auto else namespace.batch_size

    datamodule = SASDataModule(data_dir=data_dir,
                               sub_dir=namespace.sub_dir,
                               batch_size=batch_size,
                               n_bins=namespace.n_bins,
                               val_size=namespace.val_size,
                               subsample=namespace.subsample,
                               seed=namespace.seed)
    datamodule.setup()  # needed to initialze num_reg, num_clf and scalers

    # initialize model and trainer
    if namespace.disable_logger:
        logger = None
    else:
        logger = WandbLogger(project=namespace.project_name,
                             save_dir=os.path.join(
                                 root_dir, namespace.log_dir),
                             log_model='all')

    if namespace.from_yaml is not None:
        params = load_hparams_from_yaml(namespace.from_yaml)
    else:
        params = load_hparams_from_namespace(namespace)
    params['input_transformer'] = datamodule.input_transformer
    params['target_transformer'] = datamodule.target_transformer

    model = SASPerceiverIOModel(datamodule.num_clf,
                                datamodule.num_reg,
                                **params)

    if namespace.batch_size_auto:
        # estimate batch size
        batch_size = estimate_batch_size(model, datamodule)
        params['batch_size'] = batch_size
        datamodule.batch_size = batch_size
        # annoying but necessary for correct wandb batch size logging
        model.__init__(datamodule.num_clf,
                       datamodule.num_reg,
                       **params)

    strategy = DDPStrategy(
        find_unused_parameters=False) if namespace.strategy == 'ddp' else namespace.strategy
    ckpt_callback = ModelCheckpoint(
        save_top_k=0, every_n_epochs=25)  # save_top_k=-1 for every epoch
    log_every_n_steps = len(datamodule.train_dataset) // params['batch_size']
    trainer = pl.Trainer(
        gpus=namespace.gpus,
        max_epochs=namespace.max_epochs,
        gradient_clip_val=namespace.gradient_clip_val,
        logger=logger,
        precision=32,
        callbacks=[ckpt_callback],
        accumulate_grad_batches=namespace.accumulate_grad_batches,
        overfit_batches=namespace.overfit_batches,
        deterministic=bool(namespace.deterministic),
        strategy=strategy,
        num_nodes=namespace.num_nodes,
        detect_anomaly=bool(namespace.detect_anomaly),
        log_every_n_steps=log_every_n_steps,
        flush_logs_every_n_steps=1e12  # this prevents training from freezing at 100 steps
    )
    trainer.fit(model,
                datamodule=datamodule,
                ckpt_path=namespace.ckpt_path)
