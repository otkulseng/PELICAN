import tensorflow as tf
from keras import layers

class Mask(layers.Layer):
    def __init__(self, n_obj,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_obj = n_obj

    def build(self, input_shape):
        #input shape = batch x N x N x L
        pass
    def make(self, inputs):
        particle_sum = tf.reduce_sum(tf.abs(inputs), axis=-1)
        self.mask = tf.expand(tf.cast(particle_sum > 1.0e-6, tf.int32), axis=-1)
        return self

    def call(self, x):
        return x * self.mask