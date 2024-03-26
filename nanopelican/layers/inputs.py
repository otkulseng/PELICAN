import tensorflow as tf

from keras.layers import Layer
from nanopelican import data

class DataHandler(Layer):
    def __init__(self,data_format='fourvec', num_particles=32, **kwargs):
        super().__init__(trainable=False, **kwargs)

        self.data_handler = data.get_handler(data_format)
        self.num_particles=num_particles

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs):
        # Assumes input_shape is
        # Batch x num_particles (padded) x CUSTOM
        # where self.data_handler is supposed to convert
        # to the inner products Batch x num_particles x num_particles
        # inputs = inputs[..., :5, :]
        inputs = inputs[...,:self.num_particles, :]

        # Quantize Bits Here!!



        inner_prods = self.data_handler(inputs)

        return tf.reshape(tf.expand_dims(inner_prods, -1), (-1, self.num_particles, self.num_particles, 1))



