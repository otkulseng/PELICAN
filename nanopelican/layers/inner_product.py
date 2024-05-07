import tensorflow as tf

from keras import layers
import logging

from .util import get_handler, get_instantons, repeat_const



class InnerProduct(layers.Layer):
    def __init__(self, arg_dict):
        super().__init__()
        self.arg_dict = arg_dict
        self.data_handler = get_handler(arg_dict['data_format'])

        self.logger = logging.getLogger('')
        logging.warning("Instantons do not work here yet")
        self.use_instantons = False
        if self.use_instantons:
            self.instantons = get_instantons(arg_dict['data_format'])
            self.instantons = tf.expand_dims(self.instantons, axis=0)


    def build(self, input_shape):
        return super().build(input_shape)

    def compute_output_shape(self, input_shape):
        # if inp is (B, N, custom)
        N = input_shape[-2]

        if self.use_instantons:
            return (None, N+len(self.instantons), N+len(self.instantons), 1)

        return (None, N, N, 1)

    def call(self, inputs):
        # Assumes input_shape is
        # Batch x num_particles (padded) x CUSTOM
        # where self.data_handler is supposed to convert
        # to the inner products Batch x num_particles x num_particles
        # # Add instantons

        if self.use_instantons:
            instantons = layers.Lambda(lambda x: repeat_const(x, self.instantons))(inputs)
            inputs = layers.Concatenate(axis=-2)([inputs, instantons])


        # inputs[..., -1, :] = tf.constant([1, 0, 0, 1])
        # inputs[..., -2, :] = tf.constant([1, 0, 0, -1])

        # TODO: Quantize Bits Here!!

        inner_prods = self.data_handler(inputs)

        return tf.expand_dims(inner_prods, axis=-1)


