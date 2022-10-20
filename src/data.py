import copy
import json
import math
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, StandardScaler
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Optional


def load_param_names(fp):
    assert fp.endswith('_par_names.json')
    with open(fp, 'r') as f:
        param_names = json.load(f)
    return param_names


def raw_data_to_df(data_dir: str, step: int = 1):
    fns = sorted(os.listdir(data_dir))
    model_names = [x.split('_data.npy')[0]
                   for x in fns if x.endswith('data.npy')]

    model_dfs = []

    for model_idx, model_name in enumerate(model_names):
        model_data = np.load(os.path.join(data_dir, f'{model_name}_data.npy'))
        param_names = load_param_names(os.path.join(
            data_dir, f'{model_name}_par_names.json'))
        params = np.load(os.path.join(data_dir, f'{model_name}_pars.npy'))
        q_values = np.load(os.path.join(
            data_dir, f'{model_name}_q_values.npy'))[::step]

        x_columns = [f'I(q={q})' for q in q_values]
        x_df = pd.DataFrame(model_data, columns=x_columns)
        # clf targets
        x_df['model'] = [model_name]*x_df.shape[0]
        x_df['model_label'] = [model_idx]*x_df.shape[0]
        # reg targets
        if len(param_names) == 0:
            y_reg_df = pd.DataFrame()
        else:
            y_columns = [
                f'reg-model={model_name}-param={x}' for x in param_names]
            y_reg_df = pd.DataFrame(params, columns=y_columns)
        model_df = pd.concat([x_df, y_reg_df], axis=1)
        # filters
        model_df = model_df[model_df.columns.drop(
            list(model_df.filter(regex='polydispersity')))]
        model_df = model_df[model_df.columns.drop(
            list(model_df.filter(regex='sld')))]

        model_dfs.append(model_df)

    return pd.concat(model_dfs, axis=0, ignore_index=True)


def quotient_transform(x):
    return x[:, 1:] / x[:, :-1]


def scalar_neutralization(x):
    x_qt = quotient_transform(x)
    return np.cumprod(x_qt, axis=-1)


class IqTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=256):
        self.n_bins = n_bins
        self.input_transform = FunctionTransformer(
            self.input_transform_)
        self.discretizer = KBinsDiscretizer(
            n_bins,
            encode='ordinal',
            strategy='quantile',
            subsample=None)

    def fit(self, x):
        x = self.input_transform.transform(x)
        # self.discretizer.fit(x)
        self.discretizer.fit(np.reshape(x, (-1, 1)))
        return self

    def transform(self, x):
        x = self.input_transform.transform(x)
        # x = self.discretizer.transform(x)
        x = np.reshape(self.discretizer.transform(
            np.reshape(x, (-1, 1))), (-1, x.shape[-1]))
        return x

    def input_transform_(self, x):
        return np.log(quotient_transform(x**2))
        # return np.log(scalar_neutralization(x**2))
        # return np.log(x**2) / np.max(np.log(x**2), axis=0, keepdims=True)


class TargetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.log_indices = None
        self.scaler = StandardScaler()

    def fit(self, x):
        x = copy.deepcopy(x)
        n, f = x.shape
        log_indices = np.array(
            [i for i in range(f) if not self.check_col_is_uniform_(x[:, i])], dtype=int)
        self.log_indices = log_indices
        x[:, log_indices] = np.log(x[:, log_indices])
        self.scaler.fit(x)
        return self

    def transform(self, x):
        x = copy.deepcopy(x)
        x[:, self.log_indices] = np.log(x[:, self.log_indices])
        return self.scaler.transform(x)

    def inverse_transform(self, x):
        x = copy.deepcopy(x)
        x = self.scaler.inverse_transform(x)
        x[:, self.log_indices] = np.exp(x[:, self.log_indices])
        return x

    def check_col_is_uniform_(self, y_i):
        y_i = y_i[~np.isnan(y_i)]
        y_i = y_i - y_i.min()
        qs = np.linspace(0, 1, 100)
        y_i_quantiles = np.quantile(y_i, qs)

        u_i = np.random.uniform(y_i.min(), y_i.max(), size=(len(y_i)))
        u_i_quantiles = np.quantile(u_i, qs)

        y_norm = np.linalg.norm(y_i_quantiles)
        u_norm = np.linalg.norm(u_i_quantiles)
        dist = np.dot(y_i_quantiles, u_i_quantiles) / (y_norm*u_norm)
        return math.isclose(dist, 1.0, abs_tol=0.01)


class SASDataset:
    def __init__(self, df: pd.DataFrame,
                 x_scaler: IqTransformer,
                 y_scaler: TargetTransformer):
        self.df = df.reset_index(drop=True)
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

        data_columns = [x for x in df.columns if x.startswith('I(q')]
        reg_target_columns = [x for x in df.columns if x.startswith('reg')]

        self.I_q = df[data_columns].values
        self.I_q_transformed = x_scaler.transform(df[data_columns].values)
        self.reg_targets = df[reg_target_columns].values
        self.reg_targets_transformed = self.y_scaler.transform(
            self.reg_targets)
        self.clf_labels = df['model_label'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # I_q = self.I_q_transformed[idx, :, None]
        I_q = self.x_scaler.transform(self.I_q[idx, :][None, :]).T
        reg_targets = self.reg_targets_transformed[idx, :]
        return torch.LongTensor(I_q), self.clf_labels[idx], torch.Tensor(reg_targets)


class SASDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 sub_dir: str,
                 batch_size: int,
                 val_size: float = 0.0,
                 n_bins: int = 256,
                 subsample: int = None,
                 seed: int = None):
        super().__init__()
        self.data_dir = data_dir
        self.sub_dir = sub_dir
        self.batch_size = batch_size
        self.val_size = val_size
        self.n_bins = n_bins
        self.subsample = subsample
        self.seed = seed
        self.num_clf = None
        self.num_reg = None
        self.input_transformer = None
        self.target_transformer = None

    def setup(self, stage: Optional[str] = None):
        train = pd.read_parquet(os.path.join(
            self.data_dir, self.sub_dir, 'train.parquet'))
        if self.subsample is not None:
            train = train.sample(n=self.subsample, random_state=self.seed)
        test = pd.read_parquet(os.path.join(
            self.data_dir, self.sub_dir, 'test.parquet'))

        self.num_clf = len(np.unique(train['model']))
        self.num_reg = len(
            [x for x in train.columns if x.startswith('reg')])

        if self.val_size > 0.0:
            train, val = train_test_split(train,
                                          test_size=self.val_size,
                                          stratify=train['model_label'],
                                          random_state=self.seed)

        # calculate I(q) and target transformers
        Iq_cols = [x for x in train.columns if x.startswith('I(q')]
        input_transformer = IqTransformer(self.n_bins)
        input_transformer.fit(train[Iq_cols].values)
        self.input_transformer = input_transformer

        reg_target_cols = [x for x in train.columns if x.startswith('reg')]
        target_transformer = TargetTransformer()
        target_transformer.fit(train[reg_target_cols].values)
        self.target_transformer = target_transformer

        # PyTorch dataset classes
        self.train_dataset = SASDataset(
            train, input_transformer, target_transformer)

        if self.val_size > 0.0:
            self.val_dataset = SASDataset(
                val, input_transformer, target_transformer)

        self.test_dataset = SASDataset(
            test, input_transformer, target_transformer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size) if self.val_size > 0.0 else None

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        raise NotImplementedError

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        ...
