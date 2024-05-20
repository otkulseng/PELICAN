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


    x_data = dataset.x_data[0][:5]

    pt = x_data[..., 0]
    eta = x_data[..., 1]
    phi = x_data[..., 2]

    E = np.cosh(eta)
    x = np.cos(phi)
    y = np.sin(phi)
    z = np.sinh(eta)

    fourvecs =(pt *  np.array([E, x, y, z])).T

    fourvecs[-1] = [1, 0, 0, -1]
    fourvecs[-2] = [1, 0, 0, 1]

    M = np.diag([1, -1, -1, -1])
    inner_prods1 = np.einsum('pi, ij, qj->pq', fourvecs, M, fourvecs)
    print(inner_prods1)


    ETA = 5e1
    PHI = np.pi / 4
    p_T = 2.0*tf.exp(-ETA)

    x_data[-2] = [p_T, ETA, PHI]
    x_data[-1] = [p_T, -ETA, PHI]
    pt = x_data[..., 0]
    eta = x_data[..., 1]
    phi = x_data[..., 2]
    pt_matr = tf.einsum('...p, ...q->...pq', pt, pt)
    eta_matr = tf.expand_dims(eta, axis=-1) - tf.expand_dims(eta, axis=-2)
    phi_matr = tf.expand_dims(phi, axis=-1) - tf.expand_dims(phi, axis=-2)
    inner_prods2 = pt_matr * (tf.cosh(eta_matr) - tf.cos(phi_matr))

    print(inner_prods2)
    print(np.linalg.norm(inner_prods1 - inner_prods2))
    assert(False)



    # Pt eta phi spurions

    ETA = 5e1
    PHI = np.pi / 4
    p_T = 2.0*tf.exp(-ETA)

    x_data[-1] = [p_T, ETA, PHI]
    x_data[-2] = [p_T, -ETA, PHI]
    pt = x_data[..., 0]
    eta = x_data[..., 1]
    phi = x_data[..., 2]

    pt_matr = tf.einsum('...p, ...q->...pq', pt, pt)
    eta_matr = tf.expand_dims(eta, axis=-1) - tf.expand_dims(eta, axis=-2)
    phi_matr = tf.expand_dims(phi, axis=-1) - tf.expand_dims(phi, axis=-2)

    # * is elementwise (hadamard)
    inner_prods = pt_matr * (tf.cosh(eta_matr) - tf.cos(phi_matr))
    print("pt eta phi")
    print(inner_prods)

def main():
    conf = load_arguments()
    return run(conf)



if __name__ == '__main__':
    main()

