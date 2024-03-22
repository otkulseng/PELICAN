import h5py
from abc import ABC
import tensorflow as tf

class DataLoader(ABC):
    def to_tfds(self):
        raise NotImplementedError("Base classes need to implement")


class H5pyLoader(DataLoader):
    def __init__(self, filename, args) -> None:
        super().__init__()
        self.file = h5py.File(filename)

        self.features  = self.file[args.feature_key]
        self.labels    = tf.reshape(self.file[args.label_key], (-1, 1))

        print(filename, self.features.shape, self.labels.shape)

    def to_tfds(self):
        features = tf.data.Dataset.from_tensor_slices(self.features)
        labels = tf.data.Dataset.from_tensor_slices(self.labels)

        return tf.data.Dataset.zip((features, labels))

