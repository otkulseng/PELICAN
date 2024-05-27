from keras import layers, Model
from nanopelican.layers import *
import tensorflow as tf


class DiagBiasDense(layers.Dense):
    def build(self, input_shape):
        super().build(input_shape)

        # B x N x N x L
        if self.use_bias:
            self.diag_bias = self.add_weight(
                name="diag_bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True
            )

    def call(self, inputs):
        N = inputs.shape[-2]
        IDENTITY = tf.eye(N, dtype=tf.dtypes.float32)
        diag_bias = tf.einsum("f, ij->ijf", self.diag_bias, IDENTITY)
        return tf.add(super().call(inputs), diag_bias)

    # def save_own_variables(self, store):
    def save_own_variables(self, store):
        super().save_own_variables(store)
        store[str(len(store))] = self.diag_bias

    def load_own_variables(self, store):
        super().load_own_variables(store)
        self.diag_bias.assign(store[str(len(store)-1)])


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

