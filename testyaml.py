#!/usr/bin/env python

# Run the training of the deepsets network given a configuration file.
import argparse
import yaml



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", required=True,  type=str)
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    print(config)

if __name__ == '__main__':
    main()

