#!/usr/bin/env python

import argparse
import yaml
import logging
from pathlib import Path
import os
import pickle
import numpy as np

import keras
from nanopelican.models import PelicanNano
from nanopelican.data import load_dataset
from nanopelican.schedulers import LinearWarmupCosineAnnealing

from tqdm.keras import TqdmCallback
from keras.optimizers import AdamW, Adam
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras import losses


import tensorflow as tf

def load_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", required=True,  type=str)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.config, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    return config




def run(conf):

    dataset = load_dataset(conf['dataset'], keys=['test']).test
    print(dataset.x_data.shape)


    pt = dataset.x_data[..., 0]
    eta = dataset.x_data[..., 1]
    phi = dataset.x_data[..., 2]

    pt_matr = tf.einsum('...p, ...q->...pq', pt, pt)
    eta_matr = tf.expand_dims(eta, axis=-1) - tf.expand_dims(eta, axis=-2)
    phi_matr = tf.expand_dims(phi, axis=-1) - tf.expand_dims(phi, axis=-2)

    # * is elementwise (hadamard)
    inner_prods = pt_matr * (tf.cosh(eta_matr) - tf.cos(phi_matr))
    print(inner_prods.shape)

def main():
    conf = load_arguments()
    return run(conf)



if __name__ == '__main__':
    main()

