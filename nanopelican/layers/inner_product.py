import tensorflow as tf

from keras import layers
import logging

from .util import get_handler, get_spurions, repeat_const, get_flops



class InnerProduct(layers.Layer):
    def __init__(self, arg_dict):
        super().__init__()
        self.arg_dict = arg_dict
        self.data_handler = get_handler(arg_dict['data_format'])
        self.use_spurions = arg_dict['spurions']

        if self.use_spurions:
            self.spurions = get_spurions(arg_dict['data_format'])
            self.spurions = tf.expand_dims(self.spurions, axis=0)

    def get_flops(self, input_shape):
        # assumes num_particles x num_features
        N, F = input_shape
        if self.use_spurions:
            N += len(self.spurions)

        return get_flops(self.arg_dict['data_format'])(input_shape)


    def build(self, input_shape):
        return super().build(input_shape)

    def compute_output_shape(self, input_shape):
        # if inp is (B, N, custom)
        N = input_shape[-2]

        if self.use_spurions:
            return (None, N+self.spurions.shape[-2], N+self.spurions.shape[-2], 1)

        return (None, N, N, 1)

    def call(self, inputs):
        # Assumes input_shape is
        # Batch x num_particles (padded) x CUSTOM
        # where self.data_handler is supposed to convert
        # to the inner products Batch x num_particles x num_particles
        # # Add spurions

        if self.use_spurions:
            spurions = layers.Lambda(lambda x: repeat_const(x, self.spurions))(inputs)
            inputs = layers.Concatenate(axis=-2)([inputs, spurions])

        # TODO: Quantize Bits Here!!

        inner_prods = self.data_handler(inputs)


        return tf.expand_dims(inner_prods, axis=-1)


