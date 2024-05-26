from pathlib import Path
import os
import yaml
import numpy as np

def add_arguments_from_dict(parser, dikt):
    for k, v, in dikt.items():
        if isinstance(v, dict):
            parser = add_arguments_from_dict(parser, v)
            continue
        parser.add_argument(f'--{k}', type=type(v))
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
        if attr is None:
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

    os.mkdir(folder)
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