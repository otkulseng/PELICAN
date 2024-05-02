import tensorflow as tf

from nanopelican import data
from keras import layers

def repeat_const(tensor, myconst):
    # https://stackoverflow.com/questions/68345125/how-to-concatenate-a-tensor-to-a-keras-layer-along-batch-without-specifying-bat
    shapes = tf.shape(tensor)
    return tf.repeat(myconst, shapes[0], axis=0)

class InnerProduct(layers.Layer):
    def __init__(self, arg_dict):
        super().__init__()
        self.arg_dict = arg_dict
        self.data_handler = data.get_handler(arg_dict['data_format'])
        self.use_instantons = arg_dict.get('instantons', False)

        if self.use_instantons:
            self.instantons = data.get_instantons(arg_dict['data_format'])
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



