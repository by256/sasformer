import os
import socket
import argparse
import numpy as np
import pandas as pd
from typing import Union
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from model import LightningModel
from data import log_relevant_regression_targets, get_scalers, SASDataset
from perceiver_io import PerceiverEncoder, PerceiverDecoder, SASPerceiverIO, TaskDecoder


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # from https://github.com/PyTorchLightning/pytorch-lightning/issues/4420
    # os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_DEBUG'] = 'INFO'

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/', type=str,
                        help='Directory containing data files.', metavar='data_dir')
    parser.add_argument('--sub_dir', default='large', type=str,
                        help='Directory containing parquet files within data_dir.', metavar='sub_dir')
    parser.add_argument('--val_size', default=0.25, type=float,
                        help='Proportion of data to split for validation', metavar='val_size')
    parser.add_argument('--log_dir', default='../logs/', type=str,
                        help='Logging directory for Tensorboard', metavar='log_dir')
    # encoder args
    parser.add_argument('--latent_dim', default=256,
                        type=int, metavar='latent_dim')
    parser.add_argument('--encoder_num_self_attn_per_block', default=4,
                        type=int, metavar='encoder_num_self_attn_per_block')
    parser.add_argument('--encoder_num_self_attn_heads', default=2,
                        type=int, metavar='encoder_num_self_attn_heads')
    # model (clf) decoder args
    parser.add_argument('--model_decoder_num_heads', default=1,
                        type=int, metavar='model_decoder_num_heads')
    # param (reg) decoder args
    parser.add_argument('--param_decoder_num_heads', default=2,
                        type=int, metavar='param_decoder_num_heads')
    # lightning model args
    parser.add_argument('--clf_weight', default=1.0,
                        type=float, metavar='clf_weight')
    parser.add_argument('--reg_weight', default=1.0,
                        type=float, metavar='reg_weight')
    # trainer args
    parser.add_argument('--batch_size', default=1024,
                        type=int, metavar='batch_size')
    parser.add_argument('--lr', default=5e-4, type=float, metavar='lr')
    parser.add_argument('--max_epochs', default=1000,
                        type=int, metavar='max_epochs')
    # parser.add_argument('--accelerator', default='gpu',
    #                     type=str, metavar='accelerator')
    parser.add_argument('--gpus', default=1, type=int, metavar='gpus')
    parser.add_argument('--accumulate_grad_batches', default=1,
                        type=int, metavar='accumulate_grad_batches')
    parser.add_argument('--overfit_batches', default=0.0,
                        type=Union[float, int], metavar='overfit_batches')
    parser.add_argument('--deterministic', default=True,
                        type=bool, metavar='deterministic')
    parser.add_argument('--strategy', default=None, type=str,
                        help='Set to `ddp` for cluster training', metavar='strategy')
    parser.add_argument('--num_nodes', default=1, type=int,
                        help='N nodes for distributed training.', metavar='num_nodes')
    parser.add_argument('--seed', default=None, type=int,
                        help='Random seed.', metavar='seed')
    namespace = parser.parse_args()

    # define paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, namespace.data_dir)

    # load data (and split if necessary)
    train = pd.read_parquet(os.path.join(
        data_dir, namespace.sub_dir, 'train.parquet'))
    num_clf = len(np.unique(train['model']))
    num_reg = len([x for x in train.columns if x.startswith('reg')])

    train = log_relevant_regression_targets(train, data_dir)
    if namespace.val_size > 0.0:
        train, val = train_test_split(train, test_size=namespace.val_size,
                                      stratify=train['model_label'], random_state=namespace.seed)

    # calculate input and output scalers
    Iq_scaler, reg_target_scaler = get_scalers(train)

    # PyTorch dataset class and loaders
    train_dataset = SASDataset(
        train, noise=False, x_scaler=Iq_scaler, y_scaler=reg_target_scaler)
    train_loader = DataLoader(
        train_dataset, batch_size=namespace.batch_size, shuffle=True, num_workers=0)
    if namespace.val_size > 0.0:
        val_dataset = SASDataset(
            val, noise=False, x_scaler=Iq_scaler, y_scaler=reg_target_scaler)
        val_loader = DataLoader(
            val_dataset, batch_size=namespace.batch_size, num_workers=0)

    # initialize model and trainer
    logger = pl.loggers.TensorBoardLogger(os.path.join(
        root_dir, namespace.log_dir), default_hp_metric=False)

    encoder = PerceiverEncoder(num_latents=namespace.latent_dim,
                               latent_dim=namespace.latent_dim,
                               input_dim=1,
                               num_self_attn_per_block=namespace.encoder_num_self_attn_per_block,
                               num_self_attn_heads=namespace.encoder_num_self_attn_heads)
    sas_model_decoder = TaskDecoder(latent_dim=namespace.latent_dim,
                                    num_heads=namespace.model_decoder_num_heads,
                                    qk_out_dim=namespace.latent_dim//4,
                                    num_outputs=num_clf)
    sas_param_decoder = TaskDecoder(latent_dim=namespace.latent_dim,
                                    num_heads=namespace.param_decoder_num_heads,
                                    qk_out_dim=namespace.latent_dim,
                                    num_outputs=num_reg)

    perceiver = SASPerceiverIO(encoder, sas_model_decoder, sas_param_decoder)
    model = LightningModel(perceiver, num_clf, lr=namespace.lr,
                           clf_weight=namespace.clf_weight, reg_weight=namespace.reg_weight)

    strategy = DDPStrategy(
        find_unused_parameters=False) if namespace.strategy == 'ddp' else namespace.strategy
    trainer = pl.Trainer(gpus=namespace.gpus,
                         max_epochs=namespace.max_epochs,
                         logger=logger,
                         precision=16,
                         accumulate_grad_batches=namespace.accumulate_grad_batches,
                         deterministic=namespace.deterministic,
                         strategy=strategy,
                         num_nodes=namespace.num_nodes,
                         flush_logs_every_n_steps=1e12  # this prevents training from freezing at 100 steps
                         )
    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader if namespace.val_size > 0 else None)
