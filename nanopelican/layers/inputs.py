import tensorflow as tf

from keras.layers import Layer
from nanopelican import data

class InnerProduct(Layer):
    def __init__(self, arg_dict):
        super().__init__()
        self.arg_dict = arg_dict
        self.data_handler = data.get_handler(arg_dict['data_format'])

    def build(self, input_shape):
        return super().build(input_shape)

    def compute_output_shape(self, input_shape):
        # if inp is (B, N, custom)
        N = input_shape[-2]
        return (None, N, N, 1)

    def call(self, inputs):
        # Assumes input_shape is
        # Batch x num_particles (padded) x CUSTOM
        # where self.data_handler is supposed to convert
        # to the inner products Batch x num_particles x num_particles
        # inputs = inputs[..., :5, :]

        # # Add instantons

        # inputs[..., -1, :] = tf.constant([1, 0, 0, 1])
        # inputs[..., -2, :] = tf.constant([1, 0, 0, -1])

        # TODO: Quantize Bits Here!!

        inner_prods = self.data_handler(inputs)

        return tf.expand_dims(inner_prods, axis=-1)



