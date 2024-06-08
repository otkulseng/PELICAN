
import argparse
from pathlib import Path
from .util import *
from .test import test
from .train import train


def ArgumentParser(filename):
    config = load_arguments(filename)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--rerun", type=str)
    parser.add_argument("--name", default='', type=str)
    parser = add_arguments_from_dict(parser, config)
    args = parser.parse_args()

    config = override_conf_from_parser(config, args)

    return config, args



def run(model):
    filename='config.yml'
    conf, args = ArgumentParser(filename)

    if args.rerun is not None:
        for file in Path.cwd().iterdir():
            if args.rerun not in file.name:
                continue
            filename = file / filename
            conf = load_arguments(filename)

            print(f"\n\nRunning train for {filename}\n\n")
            train(model, conf)
        return

    if args.test:
        print("Running test")
        test(model, args)
    else:
        print("Running train")
        train(model, conf)


