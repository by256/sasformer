import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error
import torch
import yaml

from model import SASPerceiverIOModel
from data import SASDataModule


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/', type=str,
                        help='Directory containing data files.', metavar='data_dir')
    parser.add_argument('--sub_dir', default='large', type=str,
                        help='Directory containing parquet files within data_dir.', metavar='sub_dir')
    parser.add_argument('--project_name', default='sas-perceiver', type=str,
                        help='Project name for logging.', metavar='project_name')
    parser.add_argument('--results_dir', default='../results/', type=str,
                        help='Results directory', metavar='results_dir')
    parser.add_argument('--ckpt_path', default=None, type=str,
                        help='Checkpoint path', metavar='ckpt_path')
    parser.add_argument('--batch_size', default=1024,
                        type=int, metavar='batch_size')
    parser.add_argument('--accelerator', default='gpu',
                        type=str, metavar='accelerator')
    parser.add_argument('--seed', default=None, type=int,
                        help='Random seed.', metavar='seed')
    namespace = parser.parse_args()

    # define paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, namespace.data_dir)
    results_dir = os.path.join(root_dir, namespace.results_dir)

    with open(os.path.join(data_dir, 'scales.json'), 'r') as f:
        scales = json.load(f)

    device = 'cuda:0' if namespace.accelerator == 'gpu' else 'cpu'
    model = SASPerceiverIOModel.load_from_checkpoint(
        checkpoint_path=namespace.ckpt_path).to(device)

    datamodule = SASDataModule(data_dir=data_dir,
                               sub_dir=namespace.sub_dir,
                               n_bins=128,
                               batch_size=namespace.batch_size,
                               seed=namespace.seed)
    datamodule.setup()  # needed to initialze num_reg, num_clf and scalers

    datamodule.input_transformer = model.input_transformer
    datamodule.target_transformer = model.target_transformer

    test_loader = datamodule.test_dataloader()

    y_pred_clf = []
    y_true_clf = []
    y_pred_reg = []
    y_true_reg = []

    with torch.no_grad():
        for batch in test_loader:
            y_hat_clf, y_hat_reg = model(batch[0].to(device))
            # classification
            y_pred_clf.append(torch.argmax(
                y_hat_clf, dim=1).detach().cpu().numpy())
            y_true_clf.append(batch[1].detach().cpu().numpy())
            # regression
            y_pred_reg.append(y_hat_reg.detach().cpu().numpy())
            y_true_reg.append(batch[2].detach().cpu().numpy())

    y_pred_clf = np.concatenate(y_pred_clf)
    y_true_clf = np.concatenate(y_true_clf)

    acc = accuracy_score(y_true_clf, y_pred_clf)
    print(f'Accuracy: {acc:.3f}')

    y_pred_reg = np.concatenate(y_pred_reg)
    y_true_reg = np.concatenate(y_true_reg)

    y_pred_reg = model.target_transformer.inverse_transform(y_pred_reg)
    y_true_reg = model.target_transformer.inverse_transform(y_true_reg)

    # classification results
    C = confusion_matrix(y_true_clf, y_pred_clf)
    clf_model_names = sorted(
        list(np.unique(datamodule.test_dataset.df['model'])))
    disp = ConfusionMatrixDisplay(C, display_labels=clf_model_names)
    fig, ax = plt.subplots(figsize=(22, 22))
    disp.plot(ax=ax)
    disp.im_.colorbar.remove()
    ax.set_xticklabels(clf_model_names, rotation=90)
    plt.savefig(os.path.join(results_dir, 'confusion.pdf'),
                bbox_inches='tight', transparent=False, pad_inches=0)
    plt.close()

    # regression results
    reg_target_columns = [
        x for x in datamodule.test_dataset.df.columns if x.startswith('reg')]
    model_param = [x.split('-')[1:] for x in reg_target_columns]
    model_param = [(x[0].split('=')[-1], x[1].split('=')[-1])
                   for x in model_param]
    for idx, (model_name, param_name) in enumerate(model_param):
        pred = y_pred_reg[:, idx]
        true = y_true_reg[:, idx]
        mae = np.nanmean(np.abs(pred - true))
        if param_name in scales[model_name]:
            scale = scales[model_name][param_name]
        else:
            scale = 'linear'
        print('{}:    {}    {}    MAE: {:.5f}        {}'.format(str(idx).ljust(
            3), model_name.ljust(30), param_name.ljust(20), mae, f'Scale: {scale}'.ljust(10)))

    print(
        f'Total MAE: {mean_absolute_error(y_pred_reg.ravel(), y_true_reg.ravel()):.3f}')
