import h5py
import tensorflow as tf
import numpy as np
import math

class JetDataset(tf.keras.utils.Sequence):
    def __init__(self, x_data, y_data):
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data).reshape((-1, 1))

        assert len(self.x_data) == len(self.y_data)
        self.batched_x = self.x_data
        self.batched_y = self.y_data

        self.batched = False

        # self.on_epoch_end()  # Shuffle indices initially

    def shuffle(self):
        perm = np.random.permutation(len(self.x_data))
        self.batched_x = self.x_data[perm]
        self.batched_y = self.y_data[perm]
        return self

    def batch(self, batch_size):
        self.batched = True
        num_batches = math.floor(len(self.x_data) / batch_size)
        self.batched_x = np.array([
            self.batched_x[i*batch_size:(i+1)*batch_size] for i in range(num_batches)
        ])
        self.batched_y = np.array([
            self.batched_y[i*batch_size:(i+1)*batch_size] for i in range(num_batches)
        ])
        print(self.batched_x.shape, self.batched_y.shape)
        return self


    def __len__(self):
        return len(self.batched_x)

    def __getitem__(self, index):
        return self.batched_x[index], self.batched_y[index]

    def on_epoch_end(self):
        if self.batched:
            perm = np.random.permutation(len(self.batched_x))
            self.batched_x = self.batched_x[perm]
            self.batched_y = self.batched_y[perm]



def load_h5py(filename, args):
    file = h5py.File(filename, 'r')
    features = file[args.feature_key][..., :args.num_particles, :]
    labels = file[args.label_key]

    print(f"Loading: {filename} of size {features.shape} vs {labels.shape}")

    return JetDataset(
        x_data=features,
        y_data=labels
    )
