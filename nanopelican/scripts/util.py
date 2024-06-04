from pathlib import Path
import os
import yaml
import numpy as np
from argparse import ArgumentError
from keras import callbacks
from qkeras.utils import model_save_quantized_weights

def add_arguments_from_dict(parser, dikt):
    for k, v, in dikt.items():
        if isinstance(v, dict):
            parser = add_arguments_from_dict(parser, v)
            continue

        try:
            parser.add_argument(f'--{k}', type=type(v))
        except ArgumentError:
            continue
    return parser

def set_key(dikt, key, val):
    for k, v in dikt.items():
        if isinstance(v, dict):
            set_key(v, key, val)
            continue

        if k != key:
            continue

        dikt.update({key : val})
    return dikt

def override_conf_from_parser(conf, args):
    for arg in vars(args):
        attr = getattr(args, arg)

        if type(attr) is type(None):
            continue

        conf = set_key(conf, arg, attr)
    return conf

def create_directory(name):
    counter = 0
    while True:
        folder = Path.cwd() / f'{name}-{counter}'
        if not folder.exists():
            break
        counter += 1
    os.makedirs(str(folder), exist_ok=True)
    # os.mkdir()
    return folder

def save_config_file(filename, data):
    with open(filename, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

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
    config = load_yaml('config.yml')
    config = add_default_values(config)

    return config

class MyCustom(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        print(f'lr: {lr}')

class CustomModelCheckpoint(callbacks.Callback):
    def __init__(self, monitor, mode, filepath, weights_only=False):
        self.monitor = monitor
        self.mode = mode
        self.file = filepath
        self.weights_only = weights_only

        if self.mode == 'min':
            self.cur_best = float('inf')
        elif self.mode == 'max':
            self.cur_best = -float('inf')
        else:
            raise TypeError(f'Cannot interpret mode {mode}')

    def on_epoch_end(self, epoch, logs=None):
        newval = logs[self.monitor]
        save = False
        if self.mode == 'min':
            if newval < self.cur_best:
                save = True
        elif self.mode == 'max':
            if newval > self.cur_best:
                save = True
        if save:
            self.cur_best = newval
            if self.weights_only:
                self.model.save_weights(self.file)
            else:
                self.model.save(self.file)

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.learning_rate
    return lr