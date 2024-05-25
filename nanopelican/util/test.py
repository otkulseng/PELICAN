from .load_arguments import load_yaml
import argparse
from pathlib import Path
from nanopelican import data
from keras import losses
from keras import metrics

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name", type=str, default='')
    args = parser.parse_args()
    return args

def test(model):
    args = parse()
    for elem in Path.cwd().iterdir():
        if not elem.is_dir():
            continue

        if not args.name in elem.name:
            continue

        evaluate_model(model, elem)


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

from sklearn.metrics import roc_curve, auc, log_loss, accuracy_score
from scipy.special import softmax

def generate_auc(y_true, y_pred, ax, title):

    scores = [0]*y_pred.shape[-1]
    for n in range(y_pred.shape[-1]):
        fpr, tpr, thresholds = roc_curve(y_true[..., n], y_pred[..., n])
        scores[n] = auc(fpr, tpr)
        ax.plot(fpr, tpr)

        ax.set_aspect('equal')

    ax.set_title(title + '\n' + "|".join([str(round(score, 4)) for score in scores]))
    return scores

def evaluate_model(model, folder):
    conf = load_yaml(folder / 'config.yml')
    dataset = data.load_dataset(conf['dataset'], keys=['test']).test

    eval_dir = folder / 'eval'
    os.makedirs(eval_dir, exist_ok=True)

    # Metrics plot
    fig = generate_plot(folder / 'training.log')
    fig.savefig(eval_dir / 'metrics.pdf')

    # Evaluation
    model = model(dataset.x_data.shape[1:], conf['model'])
    model.summary(expand_nested=True)
    if model.output_shape[-1] > 1:
        loss = losses.CategoricalCrossentropy()
        loss_fn = losses.categorical_crossentropy
        metric_fn = metrics.categorical_accuracy
        from_logits=False
    else:
        loss = losses.BinaryCrossentropy(from_logits=True)
        loss_fn = losses.binary_crossentropy
        metric_fn = metrics.binary_accuracy
        from_logits = True

    model.compile(
        loss=loss,
        metrics=['accuracy'],
    )


    # AUC plot
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    axes = iter(axes)

    for file in folder.iterdir():
        if '.weights.h5' not in file.name:
            continue
        name = file.name.split('.')[0]

        model.load_weights(file)
        preds = model.predict(dataset.x_data)

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

        scores = generate_auc(y_true=y_true, y_pred=preds, ax=next(axes), title=name)

    fig.savefig(eval_dir / 'auc.pdf')

        # loss, acc = model.evaluate(dataset.batch(1000))

