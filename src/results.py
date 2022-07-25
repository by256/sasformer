import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pytorch_lightning as pl
import torch
import yaml

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
    namespace = parser.parse_args()

    # define paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, namespace.data_dir)
    results_dir = os.path.join(root_dir, namespace.results_dir)

    datamodule = SASDataModule(data_dir=data_dir,
                               sub_dir=namespace.sub_dir,
                               batch_size=namespace.batch_size)
    datamodule.setup()  # needed to initialze num_reg, num_clf and scalers

    device = 'cuda:0' if namespace.accelerator == 'gpu' else 'cpu'
    model = SASPerceiverIOModel.load_from_checkpoint(
        checkpoint_path=namespace.ckpt_path).to(device)

    Iq_scaler = model.x_scaler
    datamodule.Iq_scaler = Iq_scaler
    reg_target_scaler = model.y_scaler
    datamodule.reg_target_scaler = reg_target_scaler

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

    y_pred_reg = np.concatenate(
        y_pred_reg) * reg_target_scaler.std[None, :] + reg_target_scaler.mean[None, :]
    y_true_reg = np.concatenate(
        y_true_reg) * reg_target_scaler.std[None, :] + reg_target_scaler.mean[None, :]

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
        print('{}:    {}    {}    MAE: {:.5f}'.format(str(idx).ljust(
            3), model_name.ljust(30), param_name.ljust(20), mae))
