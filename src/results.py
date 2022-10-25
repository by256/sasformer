import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error
import torch

from model import SASPerceiverIOModel
from data import SASDataModule


def top_k_acc(true, pred, k=3):
    n = len(true)
    n_correct = np.sum(
        [np.isin(a, b) for a, b in zip(true, np.argsort(pred, axis=1)[:, ::-1][:, :k])])
    return n_correct / n


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/', type=str,
                        help='Directory containing data files.', metavar='data_dir')
    parser.add_argument('--sub_dir', default='sas-55m-20k', type=str,
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
    print(model.hparams, '\n')

    datamodule = SASDataModule(data_dir=data_dir,
                               sub_dir=namespace.sub_dir,
                               n_bins=model.hparams['n_bins'],
                               batch_size=namespace.batch_size,
                               val_size=0.25,
                               seed=namespace.seed)
    datamodule.setup()  # needed to initialze scalers

    trainer = pl.Trainer(
        gpus=1 if namespace.accelerator == 'gpu' else None,
        deterministic=True)
    # trainer.validate(model, datamodule=datamodule,
    #                  ckpt_path=namespace.ckpt_path)
    test_results = trainer.test(model, datamodule=datamodule,
                                ckpt_path=namespace.ckpt_path)[0]
    global_results = {
        'acc': [test_results['test/accuracy']],
        'mae': [test_results['test/mae']]}

    test_dataloader = datamodule.test_dataloader()

    # test targets
    true_batches = [batch for batch in test_dataloader]
    y_true_clf = torch.cat([batch[1]
                           for batch in true_batches], dim=0).cpu().numpy()
    y_true_reg = torch.cat([batch[2]
                           for batch in true_batches], dim=0).cpu().numpy()
    y_true_reg = model.target_transformer.inverse_transform(y_true_reg)

    # test predictions
    pred_batches = trainer.predict(
        model, test_dataloader, ckpt_path=namespace.ckpt_path)
    y_pred_clf = torch.cat([batch[0]
                           for batch in pred_batches], dim=0).cpu().numpy()
    y_pred_reg = torch.cat([batch[1]
                           for batch in pred_batches], dim=0).cpu().numpy()
    y_pred_reg = model.target_transformer.inverse_transform(y_pred_reg)

    acc = test_results['test/accuracy']
    top3_acc = top_k_acc(y_true_clf, y_pred_clf, k=3)
    print(f'Accuracy: {acc:.5f}    Top3 Acc: {top3_acc:.5f}')

    global_results['top3_acc'] = [top3_acc]
    global_results_path = os.path.join(results_dir, 'global.csv')
    pd.DataFrame(global_results).to_csv(global_results_path, index=False)

    # per model class metrics
    test_df = datamodule.test_dataset.df

    # clf
    inter_class_acc = {'model': [], 'acc': [], 'top3_acc': []}

    model_names = test_df['model'].unique()
    for model_name in model_names:
        model_idx = test_df[test_df['model'] == model_name].index.tolist()
        acc = accuracy_score(y_true_clf[model_idx], np.argmax(
            y_pred_clf[model_idx], axis=1))
        acc = np.round(acc, 3)
        top3_acc = top_k_acc(y_true_clf[model_idx], y_pred_clf[model_idx], k=3)
        top3_acc = np.round(top3_acc, 3)
        inter_class_acc['model'].append(model_name)
        inter_class_acc['acc'].append(acc)
        inter_class_acc['top3_acc'].append(top3_acc)
        print(f'{model_name.ljust(27)}   Acc: {acc:.5f}   Top 3 Acc: {top3_acc:.5f}')

    inter_class_acc_path = os.path.join(results_dir, 'interclass_clf.csv')
    pd.DataFrame(inter_class_acc).to_csv(inter_class_acc_path, index=False)
    print('\n')

    # reg
    inter_class_mae = {'model': [], 'param': [],  'mae': [], 'scale': []}

    reg_target_columns = [x for x in test_df.columns if x.startswith('reg')]
    model_param = [x.split('-')[1:] for x in reg_target_columns]
    model_param = [(x[0].split('=')[-1], x[1].split('=')[-1])
                   for x in model_param]
    for idx, (model_name, param_name) in enumerate(model_param):
        mae = np.nanmean(np.abs(y_pred_reg[:, idx] - y_true_reg[:, idx]))
        mae = np.round(mae, 3)
        if param_name in scales[model_name]:
            scale = scales[model_name][param_name]
        else:
            scale = 'linear'
        inter_class_mae['model'].append(model_name)
        inter_class_mae['param'].append(param_name)
        inter_class_mae['mae'].append(mae)
        inter_class_mae['scale'].append(scale)
        print(
            f'{str(idx).ljust(3)}: {model_name.ljust(27)}  {param_name.ljust(18)}   MAE: {mae:.3f}   Scale: {scale.ljust(10)}')

    inter_class_mae_path = os.path.join(results_dir, 'interclass_reg.csv')
    pd.DataFrame(inter_class_mae).to_csv(inter_class_mae_path, index=False)
    print('\n')

    # classification results
    C = confusion_matrix(y_true_clf, np.argmax(
        y_pred_clf, axis=1))
    clf_model_names = sorted(
        list(np.unique(datamodule.test_dataset.df['model'])))
    disp = ConfusionMatrixDisplay(C, display_labels=clf_model_names)
    fig, ax = plt.subplots(figsize=(16, 16))
    disp.plot(ax=ax, cmap='Blues')
    disp.im_.colorbar.remove()
    ax.set_xticklabels(clf_model_names, rotation=90)
    plt.savefig(os.path.join(results_dir, 'confusion.pdf'),
                bbox_inches='tight', transparent=False, pad_inches=0)
    plt.close()
