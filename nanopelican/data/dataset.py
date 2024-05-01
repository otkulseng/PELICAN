
import h5py
import tensorflow as tf
import numpy as np
import math
from .interpreters import get_interpreter
from .util import interleave
from pathlib import Path
import logging

def load_dataset(args, keys):
    dataset = Dataset(args['folder'],
                      feature_key=args['feature_key'],
                      label_key=args['label_key'],
                      num_particles=args['num_particles'],
                      keys=keys
                      )

    load = args.get('load', False)
    if load:
        dataset.load()

    return dataset


class Dataset:
    def __init__(self, foldername, feature_key, label_key, num_particles=-1, keys=['train', 'val', 'test']) -> None:
        self.folder = Path().cwd() / foldername

        self.data_dirs = {}


        for file in self.folder.iterdir():
            for key in keys:
                if key in file.name:
                    self.data_dirs[key] = JetDataDir(file, feature_key, label_key, num_particles=num_particles)

        if len(self.data_dirs) == 0:
            raise FileNotFoundError(f"No files found in folder {self.folder.name}")

    @property
    def train(self):
        return self.data_dirs['train']
    @property
    def test(self):
        return self.data_dirs['test']
    @property
    def val(self):
        for key, val in self.data_dirs.items():
            if 'val' in key:
                return val
        return None

    def load(self):
        for _, val in self.data_dirs.items():
            val.load()


class JetDataDir(tf.keras.utils.Sequence):
    """Takes in folder of file(s) supporting

            file.get(args.feature_key)  (the resulting file must support sorted indexing)
            file.get(args.label_key)    (the resulting file must support sorted indexing)
            folder.iterdir()

        Creates a JetDataset for each file. If shuffling, all JetDatases are shuffled
        as well as the relative order of the datasets. You can think of a JetDataDir
        as a single file that you can train on, while only loading a single batch into memory
        at a time.

        When batching, each dataset is batched to the closest multiple of batch_size under
        their respective lengths

        Acts read only on the files.

    Args:
        tf (_type_): _description_
    """
    def __init__(self, folder, feature_key, label_key, num_particles):
        self.folder = folder
        logger = logging.getLogger('')

        files = [h5py.File(filename, 'r') for filename in self.folder.iterdir()]
        for file in files:
            logger.info(f"Loading file {file} of shape {file[feature_key].shape}")

        self.datasets = [
            JetDataset(x_data=file[feature_key][:, :num_particles, ...], y_data=file[label_key][:])
            for file in files
        ]

        curlen = len(self.datasets[0])
        for elem in self.datasets:
            assert(curlen == len(elem)) # Currently, only same length files work


    @property
    def x_data(self):
        return np.concatenate([np.array(dataset.x_data) for dataset in self.datasets])

    @property
    def y_data(self):
        return np.concatenate([np.array(dataset.y_data) for dataset in self.datasets])

    def shuffle(self, keepratio=False):
        np.random.shuffle(self.datasets)
        self.datasets = [dataset.shuffle(keepratio=keepratio) for dataset in self.datasets]
        return self

    def batch(self, batch_size):
        self.datasets = [dataset.batch(batch_size) for dataset in self.datasets]
        return self

    def __len__(self):
        return np.sum([len(dataset) for dataset in self.datasets])

    def __getitem__(self, index):
        dataset_index = index // len(self.datasets[0])
        batch_index = index % len(self.datasets[dataset_index])

        return self.datasets[dataset_index][batch_index]

    def on_epoch_end(self):
        self.shuffle()

    def load(self):
        for dataset in self.datasets:
            dataset.load()

class JetDataset(tf.keras.utils.Sequence):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data


        self.logger = logging.getLogger('')
        self.logger.warning("Adding instantons here, make sure to match format")

        self.x_data[:, -1, ...] = np.array([1, 0, 0, -1])
        self.x_data[:, -2, ...] = np.array([1, 0, 0, 1])

        assert len(self.x_data) == len(self.y_data)

        self.batches = np.arange(len(x_data))
        self.batch_size=1

    def shuffle(self, keepratio=False):
        if not keepratio or self.batch_size > 1:
            # Only shuffles first axis, so works for both batched and unbatched
            self.batches = np.random.permutation(self.batches)
            return self

        self.logger.warning("Does NOT work with more classes than up down")

        # Shuffle while keeping same ratio of answers
        top = np.where(self.y_data==1)[0]
        bottom = np.where(self.y_data==0)[0]

        top = np.random.permutation(top)
        bottom = np.random.permutation(bottom)

        self.batches = interleave([top, bottom])
        return self

    def batch(self, batch_size):
        self.batch_size=batch_size

        num_batches = math.floor(len(self.x_data) / batch_size)

        true_len = num_batches * batch_size
        self.batches = self.batches.flatten()[:true_len].reshape((num_batches, -1))
        self.batches.sort(axis=-1)

        return self


    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return self.x_data[self.batches[index]], self.y_data[self.batches[index]]

    def on_epoch_end(self):
        if self.batch_size > 1:
            self.shuffle(keepratio=False)

    def load(self):
        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data)
        return self


