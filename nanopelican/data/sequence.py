from keras.utils import Sequence
import math
from sklearn.utils import shuffle
import h5py
import tensorflow as tf

class JetDataset(Sequence):
    def __init__(self, x_data, y_data, batch_size) -> None:
        self.x_data = x_data
        self.y_data = y_data

        self.x_data, self.y_data = shuffle(self.x_data, self.y_data)

        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x_data) / self.batch_size)

    def __getitem__(self, index):
        low = index * self.batch_size
        high = min(low + self.batch_size, len(self.x_data))

        return self.x_data[low:high], self.y_data[low:high]

    def on_epoch_end(self):
        self.x_data, self.y_data = shuffle(self.x_data, self.y_data)
        return super().on_epoch_end()

def load_h5py(filename, args):
    file        = h5py.File(filename)
    features    = file[args.feature_key][..., :args.num_particles, :]
    labels      = file[args.label_key][:]
    print(filename, features.shape, labels.shape)

    return JetDataset(
        x_data=features,
        y_data=labels,
        batch_size=args.batch_size
    )



