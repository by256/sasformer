import gc
import os
import wandb
import joblib
import optuna
import argparse
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers.wandb import WandbLogger

from data import SASDataModule
from train import estimate_batch_size
from model import SASPerceiverIOModel


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()
    pl.utilities.memory.garbage_collection_cuda()


def objective(trial, namespace, root_dir, data_dir):
    dropout = trial.suggest_float(
        'dropout', 0.1, 0.5, step=0.05)
    params_i = {'lr': trial.suggest_loguniform('lr', 1e-5, 1e-1),
                'batch_size': 2,   # placeholder
                'latent_dim': trial.suggest_categorical('latent_dim', [32, 64, 128, 256, 512]),
                # encoder args
                'enc_num_self_attn_per_block': trial.suggest_int('enc_num_self_attn_per_block', 2, 4),
                'enc_num_cross_attn_heads': trial.suggest_categorical('enc_num_cross_attn_heads', [1, 2, 4]),
                'enc_num_self_attn_heads': trial.suggest_categorical('enc_num_self_attn_heads', [1, 2, 4]),
                'enc_cross_attn_widening_factor': trial.suggest_int('enc_cross_attn_widening_factor', 1, 2),
                'enc_self_attn_widening_factor': trial.suggest_int('enc_self_attn_widening_factor', 1, 2),
                'enc_dropout': dropout,
                'enc_cross_attention_dropout': dropout,
                'enc_self_attention_dropout': dropout,
                # model decoder args
                'model_dec_widening_factor': trial.suggest_int('model_dec_widening_factor', 1, 2),
                'model_dec_num_heads': trial.suggest_categorical('model_dec_num_heads', [1, 2, 4]),
                'model_dec_qk_out_dim': 64,
                'model_dec_dropout': dropout,
                'model_dec_attn_dropout': dropout,
                # param decoder args
                'param_dec_widening_factor': trial.suggest_int('param_dec_widening_factor', 1, 2),
                'param_dec_num_heads': trial.suggest_categorical('param_dec_num_heads', [1, 2, 4, 8]),
                'param_dec_qk_out_dim': 256,
                'param_dec_dropout': trial.suggest_float('param_dec_dropout', 0.1, 0.5, step=0.05),
                # loss args
                'clf_weight': 1.0,
                'reg_weight': trial.suggest_loguniform('reg_weight', 1e-3, 1e1),
                }

    datamodule = SASDataModule(data_dir=data_dir,
                               sub_dir=namespace.sub_dir,
                               batch_size=32,  # place holder
                               val_size=namespace.val_size,
                               seed=namespace.seed)
    datamodule.setup()  # needed to initialze num_reg, num_clf and scalers

    model = SASPerceiverIOModel(datamodule.num_clf,
                                datamodule.num_reg,
                                x_scaler=datamodule.Iq_scaler,
                                y_scaler=datamodule.reg_target_scaler,
                                **params_i)

    batch_size = estimate_batch_size(model, datamodule)
    params_i['batch_size'] = batch_size
    datamodule.batch_size = batch_size

    # annoying but necessary for correct wandb batch size logging
    model.__init__(datamodule.num_clf,
                   datamodule.num_reg,
                   x_scaler=datamodule.Iq_scaler,
                   y_scaler=datamodule.reg_target_scaler,
                   **params_i)

    logger = WandbLogger(project=namespace.project_name,
                         save_dir=os.path.join(root_dir, namespace.log_dir))

    strategy = DDPStrategy(
        find_unused_parameters=False) if namespace.strategy == 'ddp' else namespace.strategy
    trainer = pl.Trainer(gpus=namespace.gpus,
                         max_epochs=namespace.max_epochs,
                         gradient_clip_val=namespace.gradient_clip_val,
                         logger=logger,
                         precision=16,
                         accumulate_grad_batches=namespace.accumulate_grad_batches,
                         strategy=strategy,
                         num_nodes=namespace.num_nodes,
                         detect_anomaly=False,
                         flush_logs_every_n_steps=1e12  # this prevents training from freezing at 100 steps
                         )
    trainer.fit(model,
                datamodule=datamodule)
    val_results = trainer.validate(model, datamodule=datamodule)[0]

    del logger
    del datamodule
    del model
    del strategy
    del trainer
    clear_cache()
    wandb.finish()
    return val_results['val/accuracy'], val_results['val/mae']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/', type=str,
                        help='Directory containing data files.', metavar='data_dir')
    parser.add_argument('--sub_dir', default='large', type=str,
                        help='Directory containing parquet files within data_dir.', metavar='sub_dir')
    parser.add_argument('--project_name', default='hp-search', type=str,
                        help='Project name for logging.', metavar='project_name')
    parser.add_argument('--val_size', default=0.25, type=float,
                        help='Proportion of data to split for validation', metavar='val_size')
    parser.add_argument('--log_dir', default='../logs/', type=str,
                        help='Logging directory for Tensorboard/WandB', metavar='log_dir')
    parser.add_argument('--max_epochs', default=200,
                        type=int, metavar='max_epochs')
    parser.add_argument('--gradient_clip_val', default=3.0,
                        type=float, metavar='gradient_clip_val')
    parser.add_argument('--gpus', default=1, type=int, metavar='gpus')
    parser.add_argument('--accumulate_grad_batches', default=1,
                        type=int, metavar='accumulate_grad_batches')
    parser.add_argument('--strategy', default=None, type=str,
                        help='Set to `ddp` for cluster training', metavar='strategy')
    parser.add_argument('--num_nodes', default=1, type=int,
                        help='N nodes for distributed training.', metavar='num_nodes')
    parser.add_argument('--resume', default=False, type=bool,
                        help='Resume search from checkpoint if it exists.', metavar='resume')
    parser.add_argument('--seed', default=None, type=int,
                        help='Random seed.', metavar='seed')
    namespace = parser.parse_args()

    # define paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, namespace.data_dir)

    trials_results_path = os.path.join(
        root_dir, namespace.log_dir, 'results.csv')
    study_path = os.path.join(root_dir, namespace.log_dir, 'study.pkl')

    if namespace.resume and os.path.isfile(study_path):
        study = joblib.load(study_path)
    else:
        study = optuna.create_study(directions=['maximize', 'minimize'])
    study.optimize(lambda trial: objective(trial, namespace, root_dir, data_dir),
                   n_trials=100,
                   gc_after_trial=True,
                   callbacks=[lambda study, trial: study.trials_dataframe().to_csv(trials_results_path, index=False),
                              lambda study, trial: joblib.dump(study, study_path)])
