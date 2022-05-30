import os
import json
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
import pytorch_lightning as pl


def raw_data_to_df(data_dir: str, sub_dir: str = 'large', step: int = 2) -> pd.DataFrame:

    data_file_exts = ['pars.npy', 'par_names.json', 'data.npy']
    q_values_path = os.path.join(data_dir, 'q_values.npy')
    q_values = np.load(q_values_path)[::step]
    n_q = len(q_values)

    raw_data_dir = os.path.join(data_dir, sub_dir)

    model_names = [x for x in os.listdir(
        raw_data_dir) if x.endswith('_q_values.npy')]
    model_names = sorted([x.split('_q_values.npy')[0] for x in model_names])

    data = list()

    for model_idx, model_name in enumerate(model_names):
        # I(q)
        model_I_q_fn = '{}_data.npy'.format(model_name)
        model_I_q = np.load(os.path.join(
            raw_data_dir, model_I_q_fn))[:, ::step]

        # regression targets
        param_names_fn = '{}_par_names.json'.format(model_name)
        with open(os.path.join(raw_data_dir, param_names_fn), 'rb') as f:
            param_names = json.load(f)
        param_names = ['reg-model={}-param={}'.format(
            model_name, param_name) for param_name in param_names]
        param_values_fn = '{}_pars.npy'.format(model_name)
        param_values = np.load(os.path.join(raw_data_dir, param_values_fn))

        # remove polydispersity regression targets
        if '{}_polydispersity'.format(model_name) in param_names:
            pd_mask = ['polydispersity' in x for x in param_names]
            param_names = [param_names[i]
                           for i, x in enumerate(pd_mask) if not x]
            param_values = param_values[:, np.bitwise_not(pd_mask)]

        n_model_rows = model_I_q.shape[0]
        for i in range(n_model_rows):
            I_q = {'I(q={})'.format(q_values[j])
                      : model_I_q[i, j] for j in range(n_q)}
            clf_labels = {'model': model_name, 'model_label': model_idx}
            reg_targets = {param_names[j]: param_values[i, j]
                           for j in range(len(param_names))}
            row_dict = I_q | clf_labels | reg_targets
            data.append(row_dict)

    return pd.DataFrame(data)


@dataclass
class IqScaler:
    """dataclass for storing mean and std of log(I(q)**2)"""
    mean: str
    std: float


@dataclass
class RegressionScaler:
    """dataclass for storing mean and std of regression targets"""
    mean: str
    std: float


class SASDataset:
    def __init__(self, df: pd.DataFrame, noise: bool = False, noise_scale: float = 0.01, x_scaler: IqScaler = None, y_scaler: RegressionScaler = None):
        self.df = df.reset_index(drop=True)
        self.noise = noise
        self.noise_scale = noise_scale
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

        data_columns = [x for x in df.columns if x.startswith('I(q')]
        reg_target_columns = [x for x in df.columns if x.startswith('reg')]

        self.I_q = df[data_columns].values
        self.reg_targets = df[reg_target_columns].values
        self.clf_labels = df['model_label'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        I_q = self.I_q[idx, :, None]
        reg_targets = self.reg_targets[idx, :]

        if self.noise:
            I_q = I_q + \
                np.random.normal(scale=self.noise_scale, size=I_q.shape)
        I_q = np.log10(I_q**2)
        if self.x_scaler is not None:
            # scale I_q
            I_q = (I_q - self.x_scaler.mean) / self.x_scaler.std

        if self.y_scaler is not None:
            reg_targets = (reg_targets - self.y_scaler.mean) / \
                self.y_scaler.std

        return torch.Tensor(I_q), self.clf_labels[idx], torch.Tensor(reg_targets)


def log_relevant_regression_targets(df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    reg_target_columns = [x for x in df.columns if x.startswith('reg')]

    with open(os.path.join(data_dir, 'scales.json'), 'r') as f:
        scales = json.load(f)
    model_param = [x.split('-')[1:] for x in reg_target_columns]
    model_param = [(x[0].split('=')[-1], x[1].split('=')[-1])
                   for x in model_param]
    for idx, (model_name, param_name) in enumerate(model_param):
        col_name = reg_target_columns[idx]
        if param_name == 'polydispersity':
            continue
        scale = scales[model_name][param_name]
        if scale == 'log':
            df[col_name] = np.log(df[col_name])
    return df


def get_scalers(df_train: pd.DataFrame) -> Tuple[IqScaler, RegressionScaler]:
    data_columns = [x for x in df_train.columns if x.startswith('I(q')]
    reg_target_columns = [x for x in df_train.columns if x.startswith('reg')]

    I_q_train_transformed = np.log10(
        df_train[data_columns].values**2)  # just for scaler
    I_q_mean = I_q_train_transformed.mean()  # global
    I_q_std = I_q_train_transformed.std()  # global
    Iq_scaler = IqScaler(mean=I_q_mean, std=I_q_std)
    del I_q_train_transformed

    reg_mean = np.nanmean(
        df_train[reg_target_columns].values, axis=0)  # feature-wise
    reg_std = np.nanstd(
        df_train[reg_target_columns].values, axis=0)  # feature-wise
    reg_target_scaler = RegressionScaler(mean=reg_mean, std=reg_std)
    return Iq_scaler, reg_target_scaler


class SASDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, sub_dir: str, batch_size: int, val_size: float = 0.0, seed: int = None):
        super().__init__()
        self.data_dir = data_dir
        self.sub_dir = sub_dir
        self.batch_size = batch_size
        self.val_size = val_size
        self.seed = seed
        self.num_clf = None
        self.num_reg = None
        self.Iq_scaler = None
        self.reg_target_scaler = None
        # self.setup_done = False

    def setup(self, stage: Optional[str] = None):
        # if not self.setup_done:
        train = pd.read_parquet(os.path.join(
            self.data_dir, self.sub_dir, 'train.parquet'))
        train = train.sample(n=8192)  # for debugging REMOVE THIS LATER
        test = pd.read_parquet(os.path.join(
            self.data_dir, self.sub_dir, 'test.parquet'))

        self.num_clf = len(np.unique(train['model']))
        self.num_reg = len(
            [x for x in train.columns if x.startswith('reg')])

        train = log_relevant_regression_targets(train, self.data_dir)
        if self.val_size > 0.0:
            train, val = train_test_split(train,
                                          test_size=self.val_size,
                                          stratify=train['model_label'],
                                          random_state=self.seed)

        # calculate input and output scalers
        Iq_scaler, reg_target_scaler = get_scalers(train)
        self.Iq_scaler = Iq_scaler
        self.reg_target_scaler = reg_target_scaler

        # PyTorch dataset classes
        self.train_dataset = SASDataset(
            train, noise=False, x_scaler=Iq_scaler, y_scaler=reg_target_scaler)

        if self.val_size > 0.0:
            self.val_dataset = SASDataset(
                val, noise=False, x_scaler=Iq_scaler, y_scaler=reg_target_scaler)

        self.test_dataset = SASDataset(
            test, noise=False, x_scaler=Iq_scaler, y_scaler=reg_target_scaler)

        # self.setup_done = True

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        raise NotImplementedError('predict_dataloader method not implemented.')

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        ...
