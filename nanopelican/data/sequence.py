
import h5py
import tensorflow as tf
import numpy as np
import math
from .interpreters import get_interpreter


class JetDataDir(tf.keras.utils.Sequence):
    """Takes in folder of files supporting

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
    def __init__(self, folder, args):
        self.load = args.load
        self.folder = folder

        files = [h5py.File(filename, 'r') for filename in self.folder.iterdir()]

        self.datasets = []
        interpreter = get_interpreter(args.data_interpreter)
        for file in files:
            x_data = file.get(args.feature_key)
            y_data = file.get(args.label_key)

            x_data, y_data = interpreter(x_data, y_data)

            self.datasets.append(JetDataset(x_data=x_data, y_data=y_data))


        curlen = len(self.datasets[0])
        for elem in self.datasets:
            assert(curlen == len(elem)) # Currently, only same length files work

        if self.load:
            for dataset in self.datasets:
                dataset.load()

        self.batch_size = 1

    def shuffle(self):
        np.random.shuffle(self.datasets)
        self.datasets = [dataset.shuffle() for dataset in self.datasets]
        return self

    def batch(self, batch_size):
        self.batch_size = batch_size
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



class JetDataset(tf.keras.utils.Sequence):
    def __init__(self, x_data, y_data, load=False):
        self.x_data = x_data
        self.y_data = y_data

        if load:
            self.x_data = np.array(x_data)
            self.y_data = np.array(y_data)

        assert len(self.x_data) == len(self.y_data)

        self.batches = np.arange(len(x_data))
        self.batch_size=1

    def shuffle(self):
        # Only shuffles first axis, so works for both batched and unbatched
        self.batches = np.random.permutation(self.batches)
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
            self.shuffle()

    def load(self):
        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data)
        return self


def load_h5py(filename, args):

    return JetDataDir(
        folder=filename,
        args=args
    )
