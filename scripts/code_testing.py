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



def pretty_print(flop_dict):
    total = 0
    for k, v in flop_dict.items():
        if isinstance(v, dict):
            print(f'\n - - {k} - -')
            print('-'*20)
            total += pretty_print(v)
            print('-'*20)
        else:
            print(f'{k} : {v}')
            total += v
    print(f"Total {total}")
    return total


def run(conf):

   model =  PelicanNano(conf['model'])

   data = load_dataset(conf[''])

   flops = model.get_flops((32, 3))
   pretty_print(flops)


def main():
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('--foo', type=str)
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    print(args)



if __name__ == '__main__':
    main()

