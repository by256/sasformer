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


def raw_data_to_df(data_dir: str, sub_dir: str = 'large', step: int = 2) -> pd.DataFrame:

    data_file_exts = ['pars.npy', 'par_names.json', 'data.npy']
    q_values_path = os.path.join(data_dir, 'q_values.npy')
    q_values = np.load(q_values_path)[::step]
    n_q = len(q_values)

    raw_data_dir = os.path.join(data_dir, 'raw', sub_dir)

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
            I_q = {'I(q={})'.format(q_values[j])                   : model_I_q[i, j] for j in range(n_q)}
            clf_labels = {'model': model_name, 'model_label': model_idx}
            reg_targets = {param_names[j]: param_values[i, j]
                           for j in range(len(param_names))}
            row_dict = I_q | clf_labels | reg_targets
            data.append(row_dict)

    return pd.DataFrame(data)


def quotient_transform(x):
    x = np.concatenate([x[:, 0, None], x], axis=1)
    return x[:, 1:] / x[:, :-1]


class IqTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins=256):
        self.n_bins = n_bins
        self.square_quotient_log = FunctionTransformer(
            self.square_quotient_log_transform_)
        self.discretizer = KBinsDiscretizer(
            n_bins, encode='ordinal', strategy='quantile', subsample=None)

    def fit(self, x):
        x = self.square_quotient_log.transform(x)
        self.discretizer.fit(np.reshape(x, (-1, 1)))
        return self

    def transform(self, x):
        x = self.square_quotient_log.transform(x)
        x = np.reshape(self.discretizer.transform(
            np.reshape(x, (-1, 1))), (-1, x.shape[-1]))
        return x

    def square_quotient_log_transform_(self, x):
        return np.log(quotient_transform(x**2))


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
        self.I_q_transformed = x_scaler.transform(self.I_q)
        self.reg_targets = df[reg_target_columns].values
        self.reg_targets_transformed = self.y_scaler.transform(
            self.reg_targets)
        self.clf_labels = df['model_label'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        I_q = self.I_q_transformed[idx, :, None]
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=1, pin_memory=True) if self.val_size > 0.0 else None

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=1, pin_memory=True)

    def predict_dataloader(self):
        raise NotImplementedError

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        ...
