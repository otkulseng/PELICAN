import h5py
import pathlib
import warnings
import tensorflow as tf
from nanopelican import data

def make_dataset_from_args(args, filehandlers=['train', 'val', 'test']):
    path = pathlib.Path(args.data_dir)

    files = {}

    for file in path.iterdir():
        for key in filehandlers:
            if key in file.name:
                files[key] = file
    if len(files) == 0:
        raise ValueError(f"No test, train or validation files found in folder: {path.name}")
    return read_files(files, args)


def read_files(data_dict, args):
    database = {}
    for key, file in data_dict.items():
        if '.h5' in file.name:
            database[key] = data.load_h5py(filename=file, args=args)
        else:
            raise ValueError(f"Cannot read file {file}")

    return database


