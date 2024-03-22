import h5py
import pathlib
import warnings
import tensorflow as tf
from nanopelican import data

def make_dataset_from_args(args):
    path = pathlib.Path(args.data_dir)

    files = {}

    for file in path.iterdir():
        if 'test' in file.name:
            files['test'] = file
        elif 'train' in file.name:
            files['train'] = file
        elif 'val' in file.name:
            files['val'] = file
    if len(files) == 0:
        raise ValueError(f"No test, train or validation files found in folder: {path.name}")
    return read_files(files, args)


def read_files(data_dict, args):
    database = {}
    for key, file in data_dict.items():
        if '.h5' in file.name:
            database[key] = data.H5pyLoader(filename=file, args=args).to_tfds()
        else:
            raise ValueError(f"Cannot read file {file}")

    return database


