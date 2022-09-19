import os
import wandb
import joblib
import optuna
import argparse
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import SASDataModule
from train import clear_cache, find_batch_size_one_gpu
from model import SASPerceiverIOModel


def objective(trial, namespace, root_dir, data_dir):
    enc_dropout = trial.suggest_float(
        'enc_dropout', 0.0, 0.5, step=0.05)
    model_dec_dropout = trial.suggest_float(
        'model_dec_dropout', 0.0, 0.5, step=0.05)
    param_dec_dropout = trial.suggest_float(
        'param_dec_dropout', 0.0, 0.5, step=0.05)
    params_i = {
        'n_bins': 256,  # trial.suggest_categorical('n_bins', [128, 256, 512]),
        'num_latents': trial.suggest_categorical('num_latents', [32, 48, 64, 96, 128]),
        'latent_dim': trial.suggest_categorical('latent_dim', [256, 512, 1024, 2048]),
        # encoder args
        'enc_num_blocks': trial.suggest_int('enc_num_blocks', 2, 24),
        'enc_num_self_attn_per_block': trial.suggest_int('enc_num_self_attn_per_block', 1, 4),
        'enc_num_cross_attn_heads': trial.suggest_categorical('enc_num_cross_attn_heads', [4, 8]),
        'enc_num_self_attn_heads': trial.suggest_categorical('enc_num_self_attn_heads', [4, 8]),
        'enc_cross_attn_widening_factor': trial.suggest_int('enc_cross_attn_widening_factor', 1, 2),
        'enc_self_attn_widening_factor': trial.suggest_int('enc_self_attn_widening_factor', 1, 2),
        'enc_dropout': enc_dropout,
        'enc_cross_attention_dropout': enc_dropout,
        'enc_self_attention_dropout': enc_dropout,
        # model decoder args
        'model_dec_widening_factor': trial.suggest_int('model_dec_widening_factor', 1, 2),
        'model_dec_num_heads': trial.suggest_categorical('model_dec_num_heads', [4, 8]),
        'model_dec_qk_out_dim': trial.suggest_categorical('model_dec_qk_out_dim', [128, 256, 512, 1024]),
        'model_dec_dropout': model_dec_dropout,
        'model_dec_attn_dropout': model_dec_dropout,
        # param decoder args
        'param_dec_widening_factor': trial.suggest_int('param_dec_widening_factor', 1, 2),
        'param_dec_num_heads': trial.suggest_categorical('param_dec_num_heads', [4, 8]),
        'param_dec_qk_out_dim': trial.suggest_categorical('param_dec_qk_out_dim', [128, 256, 512, 1024]),
        'param_dec_dropout': param_dec_dropout,
        'param_dec_attn_dropout': param_dec_dropout,
        # loss args
        'clf_weight': 1.0,
        'reg_weight': 1.0
    }

    datamodule = SASDataModule(data_dir=data_dir,
                               sub_dir=namespace.sub_dir,
                               n_bins=params_i['n_bins'],
                               batch_size=1,  # placeholder
                               val_size=namespace.val_size,
                               subsample=namespace.subsample,
                               seed=namespace.seed)
    datamodule.setup()  # needed to initialze num_reg, num_clf and scalers

    # find largest batch_size that fits in memory
    batch_size = find_batch_size_one_gpu(params_i, datamodule)
    batch_size = int(batch_size * 0.9)  # buffer
    params_i['batch_size'] = batch_size
    datamodule.batch_size = batch_size
    params_i['lr'] = 7.8125e-7 * batch_size / namespace.gpus  # sorcery

    model = SASPerceiverIOModel(datamodule.num_clf,
                                datamodule.num_reg,
                                input_transformer=datamodule.input_transformer,
                                target_transformer=datamodule.target_transformer,
                                **params_i)

    logger = WandbLogger(project=namespace.project_name,
                         save_dir=os.path.join(root_dir, namespace.log_dir))

    early_stopping = EarlyStopping(monitor='val/es_metric', patience=15)
    trainer = pl.Trainer(gpus=namespace.gpus,
                         max_epochs=namespace.max_epochs,
                         gradient_clip_val=namespace.gradient_clip_val,
                         logger=logger,
                         precision=32,
                         callbacks=[early_stopping],
                         overfit_batches=namespace.overfit_batches,
                         accumulate_grad_batches=namespace.accumulate_grad_batches,
                         strategy=namespace.strategy,
                         num_nodes=namespace.num_nodes,
                         detect_anomaly=False,
                         )

    trainer.fit(model,
                datamodule=datamodule)
    val_results = trainer.validate(model, datamodule=datamodule)[0]

    del logger
    del datamodule
    del model
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
    parser.add_argument('--subsample', default=None,
                        type=int, help='Subsample data (for debugging)', metavar='subsample')
    parser.add_argument('--max_epochs', default=200,
                        type=int, metavar='max_epochs')
    parser.add_argument('--gradient_clip_val', default=1.0,
                        type=float, metavar='gradient_clip_val')
    parser.add_argument('--gpus', default=1, type=int, metavar='gpus')
    parser.add_argument('--accumulate_grad_batches', default=1,
                        type=int, metavar='accumulate_grad_batches')
    parser.add_argument('--overfit_batches', default=0,
                        type=int, metavar='overfit_batches')
    parser.add_argument('--strategy', default=None, type=str,
                        help='Set to `ddp` for cluster training', metavar='strategy')
    parser.add_argument('--num_nodes', default=1, type=int,
                        help='N nodes for distributed training.', metavar='num_nodes')
    parser.add_argument('--resume', default=0, type=int,
                        help='Resume search from checkpoint if it exists.', metavar='resume')
    parser.add_argument('--seed', default=None, type=int,
                        help='Random seed.', metavar='seed')
    namespace = parser.parse_args()

    if namespace.seed is not None:
        pl.seed_everything(namespace.seed, workers=True)

    # define paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, namespace.data_dir)

    trials_results_path = os.path.join(
        root_dir, namespace.log_dir, f'{namespace.project_name}-results.csv')
    study_path = os.path.join(
        root_dir, namespace.log_dir, f'{namespace.project_name}-study.pkl')

    if namespace.resume and os.path.isfile(study_path):
        study = joblib.load(study_path)
    else:
        study = optuna.create_study(directions=['maximize', 'minimize'])
    study.optimize(lambda trial: objective(trial, namespace, root_dir, data_dir),
                   n_trials=100,
                   gc_after_trial=True,
                   callbacks=[lambda study, trial: study.trials_dataframe().to_csv(trials_results_path, index=False),
                              lambda study, trial: joblib.dump(study, study_path)])
