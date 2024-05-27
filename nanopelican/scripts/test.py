
from pathlib import Path
from nanopelican import data
from keras import losses
from keras import metrics, models

import h5py

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

from sklearn.metrics import roc_curve, auc
from collections.abc import Iterable
from .util import *


def test(model, args):
    for elem in Path.cwd().iterdir():
        if not elem.is_dir():
            continue

        if not args.name in elem.name:
            continue

        evaluate_model(elem)

def evaluate_model(folder):
    eval_dir = folder / 'eval'
    os.makedirs(eval_dir, exist_ok=True)

    data_file = generate_data(data_dir=folder, save_dir=eval_dir)
    generate_plots(data_file=data_file, save_dir=eval_dir)


def generate_plots(data_file, save_dir):

    # # Metrics plot
    # fig = metrics_plot(folder / 'training.log')
    # fig.savefig(eval_dir / 'metrics.pdf')

    if 'runtime' in data_file:
        fig = metrics_plot(data_file['runtime'])
        fig.savefig(save_dir / 'metrics.pdf')


    df = pd.read_csv(save_dir / 'result.csv', index_col=False)
    auc_files = {}
    for elem in data_file.keys():
        if 'auc' not in data_file[elem]:
            continue
        auc_files[elem] = {
            'file': data_file[elem]['auc'],
            'title': df[df['name']==elem]['auc'].to_string().split()[-1]
        }

    fig = auc_plot(auc_files)
    fig.savefig(save_dir / 'auc.pdf')

def auc_plot(files):
    fig, axes = plt.subplots(ncols=len(files), figsize=(5 * len(files), 5))

    if not isinstance(axes, Iterable):
        axes = [axes]

    for ax, (name, file) in zip(axes, files.items()):

        title = file['title']
        file = file['file']
        for n in range(file['fpr'].shape[0]):

            fpr = file['fpr'][n, :]
            tpr = file['tpr'][n, :]
            ax.plot(fpr[fpr != 0], tpr[fpr != 0])
        ax.set_title(name + f'\n{title}')
    fig.suptitle('AUC')
    return fig


def generate_data(data_dir, save_dir):
    conf = load_yaml(data_dir / 'config.yml')
    dataset = data.load_dataset(conf['dataset'], keys=['test']).test


    save_file = h5py.File(save_dir / 'data.h5', 'a')

    # Runtime data
    runtime_data = pd.read_csv(data_dir / 'training.log')
    runtime_group = save_file.require_group('runtime')
    for elem in runtime_data:
        if elem in runtime_group:
            continue
        runtime_group.create_dataset(elem, data=runtime_data[elem])


    # Data generation

    dfs = []
    for file in data_dir.iterdir():
        if '.keras' not in file.name:
            continue
        name = file.name.split('.')[0]
        print(f'Starting: {name}')

        model = models.load_model(file)
        model.summary()


        name_group = save_file.require_group(name)
        if 'preds' not in name_group:
            preds = model.predict(dataset.x_data)
            name_group.create_dataset('preds', data=preds)

        preds = name_group['preds'][:]

        y_true = np.reshape(dataset.y_data, preds.shape)


        result_dict = {
            'name': [name]
        }

        for metric in model.metrics:
            if 'compile_metric' not in metric.name:
                continue
            res = metric(y_true=y_true, y_pred=preds)
            for k, v in res.items():
                result_dict.update({
                    k: v.numpy()
                })


        loss = np.average(model.loss(
                y_true=y_true,
                y_pred=preds
            )
        )

        result_dict.update({
            'loss': loss
        })

        scores, fpr, tpr = generate_auc(y_true=y_true, y_pred=preds)
        auc_group = name_group.require_group('auc')
        if 'fpr' not in auc_group:
            auc_group.create_dataset('fpr', data=fpr)
        if 'tpr' not in auc_group:
            auc_group.create_dataset('tpr', data=tpr)


        result_dict.update({
            'auc': "|".join([str(round(score, 4)) for score in scores])
        })

        df = pd.DataFrame.from_dict(result_dict)

        dfs.append(df)
    df = pd.concat(dfs)
    df.to_csv(save_dir / 'result.csv', index=False)
    return save_file

def metrics_plot(file):
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 5))
    for key in file:
        if 'acc' in key:
            axL.plot(file[key][:], label=key)

        if 'loss' in key:
            axR.plot(file[key][:], label=key)

    axL.set_title('Accuracy')
    axL.legend(loc='lower right')

    axR.set_title('Loss')
    axR.legend()
    fig.supxlabel('Epochs')
    return fig


def pad_list(elem):
    maxlen = max([len(e) for e in elem])
    ret = np.zeros(
        shape=(len(elem), maxlen)
    )
    for i, e in enumerate(elem):
        ret[i][:len(e)] = e
    return ret

def generate_auc(y_true, y_pred):
    scores  = [None] * y_pred.shape[-1]
    fprs    = [None] * y_pred.shape[-1]
    tprs    = [None] * y_pred.shape[-1]

    for n in range(y_pred.shape[-1]):
        fpr, tpr, _ = roc_curve(y_true[..., n], y_pred[..., n])
        scores[n] = auc(fpr, tpr)
        tprs[n] = tpr
        fprs[n] = fpr


    return scores, pad_list(fprs), pad_list(tprs)

