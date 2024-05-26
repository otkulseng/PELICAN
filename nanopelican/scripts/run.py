
import argparse
from .util import *
from .test import test
from .train import train


def ArgumentParser():
    config = load_arguments()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--name", default='', type=str)
    parser = add_arguments_from_dict(parser, config)
    args = parser.parse_args()

    config = override_conf_from_parser(config, args)

    return config, args



def run(model):
    conf, args = ArgumentParser()

    if args.test:
        print("Running test")
        test(model, args)
    else:
        print("Running train")
        train(model, conf)


