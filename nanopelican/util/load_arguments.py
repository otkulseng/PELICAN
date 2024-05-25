import argparse
import os
import yaml

import numpy as np

def add_default_values(conf):
    if 'seed' not in conf:
        conf['seed'] = np.random.randint(0, 2**31)

    if 'val_size' not in conf['hyperparams']:
        conf['hyperparams'].update({'val_size': -1})

    return conf

def load_yaml(filename):
    with open(filename, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    return config


def load_arguments():
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--config", type=str, default='config.yml')
    # parser.add_argument("--gpu", type=str, default="0")
    # args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_yaml('config.yml')

    config = add_default_values(config)

    return config