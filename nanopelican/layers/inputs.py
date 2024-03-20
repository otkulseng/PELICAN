import tensorflow as tf

from keras.layers import Layer
from nanopelican import data

class DataHandler(Layer):
    def __init__(self,data_format='fourvec', **kwargs):
        super().__init__(trainable=False, **kwargs)

        self.data_handler = data.get_handler(data_format)

    def call(self, inputs):
        # Assumes input_shape is
        # Batch x num_particles (padded) x CUSTOM
        # where self.data_handler is supposed to convert
        # to the inner products Batch x num_particles x num_particles

        inner_prods = self.data_handler(inputs)
        _, N, _ = inner_prods.shape

        return tf.reshape(tf.expand_dims(inner_prods, -1), (-1, N, N, 1))



