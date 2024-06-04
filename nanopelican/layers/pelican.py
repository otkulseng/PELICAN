from keras import layers, Model, activations
from nanopelican.layers import *
import tensorflow as tf



class DiagBiasDense(layers.Layer):
    def __init__(self, units, activation=None, *args,**kwargs):
        super().__init__(*args, **kwargs)
        self.units = units
        self.activation = activations.get(activation)

    def build(self, input_shape):
        kernel_initializer="glorot_uniform"
        bias_initializer="zeros"

        # input_shape = N x N x L
        L = input_shape[-1]

        self.kernel = self.add_weight(
            name='kernel',
            shape=(L, self.units),
            initializer=kernel_initializer
        )

        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer=bias_initializer,
            trainable=True
        )

        self.diag_bias = self.add_weight(
            name="diag_bias",
            shape=(self.units,),
            initializer=bias_initializer,
            trainable=True
        )

    def call(self, inputs):
        N = inputs.shape[-2]
        IDENTITY = tf.eye(N, dtype=tf.dtypes.float32)
        diag_bias = tf.einsum("f, ij->ijf", self.diag_bias, IDENTITY)

        return self.activation(
            tf.einsum('...ijl, lf->...ijf', inputs, self.kernel)
            + self.bias + diag_bias
        )



class PelicanNano(layers.Layer):
    def __init__(self, n_hidden, n_outputs, activation=None, num_avg=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lineq2v2 = Lineq2v2(hollow=True, symmetric=True,num_avg=num_avg)
        self.hidden = DiagBiasDense(n_hidden)
        self.activation = layers.Activation(activation=activation)
        self.lineq2v0 = Lineq2v0(num_avg=num_avg)
        self.output_layer = layers.Dense(n_outputs)

    def call(self, inputs):
        x = inputs
        x = self.lineq2v2(x)
        x = self.hidden(x)
        x = self.activation(x)
        x = self.lineq2v0(x)
        x = self.output_layer(x)
        return x

