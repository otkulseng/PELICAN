from .load_arguments import load_yaml
import argparse
from pathlib import Path
from nanopelican import data
from keras import losses
from keras import metrics

import h5py

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

from sklearn.metrics import roc_curve, auc

def test(model, args):
    for elem in Path.cwd().iterdir():
        if not elem.is_dir():
            continue

        if not args.name in elem.name:
            continue

        evaluate_model(model, elem)

def evaluate_model(model, folder):
    eval_dir = folder / 'eval'
    os.makedirs(eval_dir, exist_ok=True)

    data_file = generate_data(model, data_dir=folder, save_dir=eval_dir)
    generate_plots(data_file=data_file, save_dir=eval_dir)


def generate_plots(data_file, save_dir):

    # # Metrics plot
    # fig = generate_plot(folder / 'training.log')
    # fig.savefig(eval_dir / 'metrics.pdf')
    pass

def generate_data(model, data_dir, save_dir):
    conf = load_yaml(data_dir / 'config.yml')
    dataset = data.load_dataset(conf['dataset'], keys=['test']).test


    # Compilation
    model = model(dataset.x_data.shape[1:], conf['model'])
    model.summary(expand_nested=True)
    if model.output_shape[-1] > 1:
        loss_fn = losses.categorical_crossentropy
        metric_fn = metrics.categorical_accuracy
        from_logits=False
    else:
        loss_fn = losses.binary_crossentropy
        metric_fn = metrics.binary_accuracy
        from_logits = True

    # model.compile(
    #     loss=loss,
    #     metrics=['accuracy'],
    # )


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
        if '.weights.h5' not in file.name:
            continue

        model.load_weights(file)

        name = file.name.split('.')[0]
        print(f'Starting: {name}')
        name_group = save_file.require_group(name)
        if 'preds' not in name_group:
            preds = model.predict(dataset.x_data)
            name_group.create_dataset('preds', data=preds)

        preds = name_group['preds'][:]

        y_true = np.reshape(dataset.y_data, preds.shape)

        loss = np.average(loss_fn(
            y_true=y_true,
            y_pred=preds,
            from_logits=from_logits
        ))

        acc = np.average(metric_fn(
            y_true=y_true,
            y_pred=preds,
        ))

        scores, fpr, tpr = generate_auc(y_true=y_true, y_pred=preds)
        auc_group = name_group.require_group('auc')
        if 'fpr' not in auc_group:
            auc_group.create_dataset('fpr', data=fpr)
        if 'tpr' not in auc_group:
            auc_group.create_dataset('tpr', data=tpr)

        # print(name, loss, acc, scores)

        df = pd.DataFrame(
            {
                'name': name,
                'loss': loss,
                'acc': acc,
                'auc': scores
            }
        )
        dfs.append(df)
    df = pd.concat(dfs)
    print(df)
    df.to_csv(save_dir / 'result.csv', index=False)
    return save_file

    #     scores = generate_auc(y_true=y_true, y_pred=preds, ax=next(axes), title=name)

    # fig.savefig(eval_dir / 'auc.pdf')

def generate_plot(filename):
    file = pd.read_csv(filename)
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 5))
    for key in file:
        if 'acc' in key:
            axL.plot(file[key], label=key)

        if 'loss' in key:
            axR.plot(file[key], label=key)

    axL.set_title('Accuracy')
    axL.legend(loc='lower right')

    axR.set_title('Loss')
    axR.legend()
    fig.supxlabel('Epochs')
    return fig

def generate_auc(y_true, y_pred):
    scores = [0]*y_pred.shape[-1]
    for n in range(y_pred.shape[-1]):
        fpr, tpr, _ = roc_curve(y_true[..., n], y_pred[..., n])
        scores[n] = auc(fpr, tpr)

    return scores, fpr, tpr


# def generate_auc_(y_true, y_pred, ax, title):

#     for n in range(y_pred.shape[-1]):
#         ax.plot(fpr, tpr)

#         ax.set_aspect('equal')

#     ax.set_title(title + '\n' + "|".join([str(round(score, 4)) for score in scores]))
#     return scores

