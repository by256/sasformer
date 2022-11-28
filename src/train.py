import argparse
import gc
from mpi4py import MPI
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.tuner.tuning import Tuner
import sys
import torch
from typing import Union
import wandb
import yaml

from model import SASPerceiverIOModel
from data import SASDataModule


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()
    pl.utilities.memory.garbage_collection_cuda()


def find_batch_size_one_gpu(params, datamodule):
    model = SASPerceiverIOModel(datamodule.num_clf,
                                datamodule.num_reg,
                                input_transformer=datamodule.input_transformer,
                                target_transformer=datamodule.target_transformer,
                                **params)
    trainer = pl.Trainer(gpus=1,
                         strategy=None,
                         enable_checkpointing=False,
                         deterministic=True,
                         detect_anomaly=False)
    tuner = Tuner(trainer)
    batch_size = tuner.scale_batch_size(model,
                                        datamodule=datamodule,
                                        mode='binsearch',
                                        init_val=32,
                                        max_trials=6)
    if batch_size > 2048:
        batch_size = 2048
    clear_cache()
    return batch_size


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
               'enc_cross_attn_dropout': namespace.enc_attn_dropout,
               'enc_self_attn_dropout': namespace.enc_attn_dropout,
               'model_dec_widening_factor': namespace.model_dec_widening_factor,
               'model_dec_num_heads': namespace.model_dec_num_heads,
               #    'model_dec_qk_out_dim': namespace.model_dec_qk_out_dim,
               'model_dec_dropout': namespace.model_dec_dropout,
               'model_dec_attn_dropout': namespace.model_dec_attn_dropout,
               'param_dec_widening_factor': namespace.param_dec_widening_factor,
               'param_dec_num_heads': namespace.param_dec_num_heads,
               #    'param_dec_qk_out_dim': namespace.param_dec_qk_out_dim,
               'param_dec_dropout': namespace.param_dec_dropout,
               'param_dec_attn_dropout': namespace.param_dec_attn_dropout,
               'lr': namespace.lr,
               'batch_size': namespace.batch_size,
               'weight_decay': namespace.weight_decay,
               'n_bins': namespace.n_bins,
               #    'masked': namespace.masked,
               #    'mask_proportion': namespace.mask_proportion,
               'clf_weight': namespace.clf_weight,
               'reg_weight': namespace.reg_weight,
               'reg_obj': namespace.reg_obj}
    return hparams


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/', type=str,
                        help='Directory containing data files.', metavar='data_dir')
    parser.add_argument('--sub_dir', default='sas-55m-20k', type=str,
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
    parser.add_argument('--enc_num_blocks', default=3,
                        type=int, metavar='enc_num_blocks')
    parser.add_argument('--enc_num_self_attn_per_block', default=2,
                        type=int, metavar='encoder_num_self_attn_per_block')
    parser.add_argument('--enc_num_self_attn_heads', default=2,
                        type=int, metavar='encoder_num_self_attn_heads')
    parser.add_argument('--enc_num_cross_attn_heads', default=2,
                        type=int, metavar='enc_num_cross_attn_heads')
    parser.add_argument('--enc_cross_attn_widening_factor', default=1,
                        type=int, metavar='enc_cross_attn_widening_factor')
    parser.add_argument('--enc_self_attn_widening_factor', default=1,
                        type=int, metavar='enc_self_attn_widening_factor')
    parser.add_argument('--enc_dropout', default=0.0,
                        type=float, metavar='enc_dropout')
    parser.add_argument('--enc_attn_dropout', default=0.0,
                        type=float, metavar='enc_attn_dropout')
    # model (clf) decoder args
    parser.add_argument('--model_dec_widening_factor', default=1,
                        type=int, metavar='model_dec_widening_factor')
    parser.add_argument('--model_dec_num_heads', default=5,
                        type=int, metavar='model_decoder_num_heads')
    parser.add_argument('--model_dec_dropout', default=0.0,
                        type=float, metavar='model_dec_dropout')
    parser.add_argument('--model_dec_attn_dropout', default=0.0,
                        type=float, metavar='model_dec_attn_dropout')
    # param (reg) decoder args
    parser.add_argument('--param_dec_widening_factor', default=1,
                        type=int, metavar='param_dec_widening_factor')
    parser.add_argument('--param_dec_num_heads', default=3,
                        type=int, metavar='param_decoder_num_heads')
    parser.add_argument('--param_dec_dropout', default=0.0,
                        type=float, metavar='param_dec_dropout')
    parser.add_argument('--param_dec_attn_dropout', default=0.0,
                        type=float, metavar='param_dec_attn_dropout')
    # datamodule args
    parser.add_argument('--subsample', default=None,
                        type=int, help='Subsample data (for debugging)', metavar='subsample')
    # lightning model args
    parser.add_argument('--n_bins', default=256,
                        type=int, help='n bins for input discretization.', metavar='n_bins')
    parser.add_argument('--masked', default=1,
                        type=int, help='option to randomly mask I(q) beyond certain q index.', metavar='masked')
    parser.add_argument('--mask_proportion', default=0.23,
                        type=int, help='proportion of I(q) masked.', metavar='mask_proportion')
    parser.add_argument('--clf_weight', default=1.0,
                        type=float, metavar='clf_weight')
    parser.add_argument('--reg_weight', default=1.0,
                        type=float, metavar='reg_weight')
    parser.add_argument('--reg_obj', default='mse',
                        type=str, metavar='reg_obj')
    # lightning trainer args
    parser.add_argument('--batch_size', default=2048,
                        type=int, metavar='batch_size')
    parser.add_argument('--batch_size_auto', default=0,
                        type=int, metavar='batch_size_auto')
    parser.add_argument('--lr_auto', default=0, type=int, metavar='lr_auto')
    parser.add_argument('--lr', default=1.6e-3, type=float, metavar='lr')
    parser.add_argument('--weight_decay', default=1e-7,
                        type=float, metavar='weight_decay')
    parser.add_argument('--max_epochs', default=200,
                        type=int, metavar='max_epochs')
    parser.add_argument('--gradient_clip_val', default=1.0,
                        type=float, metavar='gradient_clip_val')
    parser.add_argument('--ckpt_path', default=None, type=str,
                        help='Checkpoint path to resume training', metavar='ckpt_path')
    # parser.add_argument('--gpus', default=1, type=int, metavar='gpus')
    parser.add_argument('--accelerator', default='gpu', type=str, metavar='accelerator')
    parser.add_argument('--devices', default=1, type=int, metavar='devices')
    parser.add_argument('--accumulate_grad_batches', default=None,
                        type=int, metavar='accumulate_grad_batches')
    parser.add_argument('--overfit_batches', default=0,
                        type=int, metavar='overfit_batches')
    parser.add_argument('--detect_anomaly', default=1,
                        type=int, metavar='detect_anomaly')
    parser.add_argument('--profile', default=0,
                        type=int, metavar='profile')
    parser.add_argument('--deterministic', default=1,
                        type=int, metavar='deterministic')
    parser.add_argument('--strategy', default=None, type=str,
                        help='Set to `ddp` for cluster training', metavar='strategy')
    parser.add_argument('--num_nodes', default=1, type=int,
                        help='N nodes for distributed training.', metavar='num_nodes')
    parser.add_argument('--seed', default=None, type=int,
                        help='Random seed.', metavar='seed')
    namespace = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if namespace.seed is not None:
        pl.seed_everything(namespace.seed, workers=True)

    # define paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, namespace.data_dir)

    datamodule = SASDataModule(data_dir=data_dir,
                               sub_dir=namespace.sub_dir,
                               batch_size=namespace.batch_size,
                               n_bins=namespace.n_bins,
                               masked=namespace.masked,
                               mask_proportion=namespace.mask_proportion,
                               val_size=namespace.val_size,
                               subsample=namespace.subsample,
                               seed=namespace.seed)
    datamodule.setup()  # needed to initialze scalers

    if namespace.from_yaml is not None:
        params = load_hparams_from_yaml(namespace.from_yaml)
    else:
        params = load_hparams_from_namespace(namespace)

    if namespace.batch_size_auto:
        batch_size = find_batch_size_one_gpu(params, datamodule)  # // 2
        batch_size = int(batch_size * 0.9)  # buffer
        params['batch_size'] = batch_size
        datamodule.batch_size = batch_size

    if namespace.lr_auto:
        params['lr'] = 0.5 * 7.8125e-7 * params['batch_size']

    params['input_transformer'] = datamodule.input_transformer
    params['target_transformer'] = datamodule.target_transformer
    params['lr'] = params['lr'] * namespace.devices * namespace.num_nodes

    # initialize model and trainer
    if not namespace.disable_logger:
        logger = WandbLogger(project=namespace.project_name,
                             save_dir=os.path.join(
                                 root_dir, namespace.log_dir),
                             log_model=False)
    else:
        logger = None

    model = SASPerceiverIOModel(datamodule.num_clf,
                                datamodule.num_reg,
                                **params)

    ckpt_callback = ModelCheckpoint(
        monitor='val/total_loss', save_top_k=1, save_last=True)
    profiler = SimpleProfiler(filename='profile') if bool(
        namespace.profile) else None

    strategy = namespace.strategy
    num_nodes = namespace.num_nodes
    devices = namespace.devices

    trainer = pl.Trainer(
        accelerator=namespace.accelerator,
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        max_epochs=namespace.max_epochs,
        gradient_clip_val=namespace.gradient_clip_val,
        logger=logger,
        precision=32,
        callbacks=[ckpt_callback],
        accumulate_grad_batches=namespace.accumulate_grad_batches,
        overfit_batches=namespace.overfit_batches,
        deterministic=bool(namespace.deterministic),
        detect_anomaly=bool(namespace.detect_anomaly),
        profiler=profiler
    )
    trainer.fit(model,
                datamodule=datamodule,
                ckpt_path=namespace.ckpt_path)
