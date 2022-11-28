import argparse
import joblib
import numpy as np
import optuna
import os
from pprint import pprint
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import wandb

from data import SASDataModule
from train import clear_cache, find_batch_size_one_gpu
from model import SASPerceiverIOModel


def objective(trial, namespace, root_dir, data_dir):
    enc_dropout = trial.suggest_float('enc_dropout', 0.0, 0.5, step=0.05)
    enc_attn_dropout = trial.suggest_float('enc_attn_dropout', 0.0, 0.5, step=0.05)
    model_dec_dropout = trial.suggest_float('model_dec_dropout', 0.0, 0.5, step=0.05)
    model_dec_attn_dropout = trial.suggest_float('model_dec_attn_dropout', 0.0, 0.5, step=0.05)
    param_dec_dropout = trial.suggest_float('param_dec_dropout', 0.0, 0.5, step=0.05)
    param_dec_attn_dropout = trial.suggest_float('param_dec_attn_dropout', 0.0, 0.5, step=0.05)
    params_i = {
        'n_bins': 256,  # trial.suggest_categorical('n_bins', [128, 256, 512]),
        'num_latents': trial.suggest_categorical('num_latents', [48, 64, 96, 128]),
        'latent_dim': trial.suggest_categorical('latent_dim', [256, 512]),
        # encoder args
        'enc_num_blocks': 1,  # trial.suggest_int('enc_num_blocks', 2, 12),
        'enc_num_self_attn_per_block': trial.suggest_int('enc_num_self_attn_per_block', 3, 8),
        'enc_num_cross_attn_heads': trial.suggest_categorical('enc_num_cross_attn_heads', [4, 8]),
        'enc_num_self_attn_heads': trial.suggest_categorical('enc_num_self_attn_heads', [4, 8]),
        'enc_cross_attn_widening_factor': 2,  # trial.suggest_int('enc_cross_attn_widening_factor', 1, 2),
        'enc_self_attn_widening_factor': 2,  # trial.suggest_int('enc_self_attn_widening_factor', 1, 2),
        'enc_dropout': enc_dropout,
        'enc_cross_attn_dropout': enc_attn_dropout,
        'enc_self_attn_dropout': enc_attn_dropout,
        # model decoder args
        'model_dec_widening_factor': 2,  # trial.suggest_int('model_dec_widening_factor', 1, 3),
        'model_dec_num_heads': 5, 
        'model_dec_dropout': model_dec_dropout,
        'model_dec_attn_dropout': model_dec_attn_dropout,
        # param decoder args
        'param_dec_widening_factor': 2,  # trial.suggest_int('param_dec_widening_factor', 1, 3),
        'param_dec_num_heads': 3, 
        'param_dec_dropout': param_dec_dropout,
        'param_dec_attn_dropout': param_dec_attn_dropout,
        # loss args
        'clf_weight': 1.0,
        'reg_weight': trial.suggest_categorical('reg_weight', np.logspace(-2, 2, 14)),
        'reg_obj': trial.suggest_categorical('reg_obj', ['mae', 'mse']),
        # optimizer args
        'batch_size': namespace.batch_size,
        'lr': namespace.lr,
        'weight_decay': trial.suggest_categorical('weight_decay', np.logspace(-7, -3, 15)),
    }
    pprint(params_i)


    datamodule = SASDataModule(data_dir=data_dir,
                               sub_dir=namespace.sub_dir,
                               n_bins=params_i['n_bins'],
                               masked=namespace.masked,
                               batch_size=params_i['batch_size'],  # placeholder
                               val_size=namespace.val_size,
                               subsample=namespace.subsample,
                               seed=namespace.seed)
    datamodule.setup()  # needed to initialze num_reg, num_clf and scalers

    model = SASPerceiverIOModel(datamodule.num_clf,
                                datamodule.num_reg,
                                input_transformer=datamodule.input_transformer,
                                target_transformer=datamodule.target_transformer,
                                **params_i)

    logger = WandbLogger(project=namespace.project_name,
                         save_dir=os.path.join(root_dir, namespace.log_dir))
    # logger = None
    early_stopping = EarlyStopping(monitor='val/es_metric', patience=50)

    strategy = namespace.strategy
    num_nodes = None if strategy == 'horovod' else namespace.num_nodes
    devices = None if strategy == 'horovod' else namespace.devices

    trainer = pl.Trainer(
        accelerator=namespace.accelerator,
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        max_epochs=namespace.max_epochs,
        gradient_clip_val=namespace.gradient_clip_val,
        logger=logger,
        precision=32,
        callbacks=[early_stopping],
        overfit_batches=namespace.overfit_batches,
        accumulate_grad_batches=namespace.accumulate_grad_batches,
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
    parser.add_argument('--sub_dir', default='sas-55m-20k', type=str,
                        help='Directory containing parquet files within data_dir.', metavar='sub_dir')
    parser.add_argument('--project_name', default='hp-search', type=str,
                        help='Project name for logging.', metavar='project_name')
    parser.add_argument('--val_size', default=0.25, type=float,
                        help='Proportion of data to split for validation', metavar='val_size')
    parser.add_argument('--log_dir', default='../logs/', type=str,
                        help='Logging directory for Tensorboard/WandB', metavar='log_dir')
    parser.add_argument('--subsample', default=None,
                        type=int, help='Subsample data (for debugging)', metavar='subsample')
    parser.add_argument('--max_epochs', default=100,
                        type=int, metavar='max_epochs')
    parser.add_argument('--gradient_clip_val', default=1.0,
                        type=float, metavar='gradient_clip_val')
    parser.add_argument('--masked', default=1,
                        type=int, help='option to randomly mask I(q) beyond certain q index.', metavar='masked')
    parser.add_argument('--accelerator', default='gpu', type=str, metavar='accelerator')
    parser.add_argument('--devices', default=1, type=int, metavar='devices')
    parser.add_argument('--batch_size', default=64, type=int, metavar='batch_size')
    parser.add_argument('--lr', default=1e-3, type=float, metavar='lr')
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
