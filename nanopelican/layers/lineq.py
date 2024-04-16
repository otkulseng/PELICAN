# The different linear equivariant layers
import tensorflow as tf
from keras.layers import Layer, BatchNormalization, Dropout
from keras import activations
import keras.backend as K
import keras

class LinearEquivariant(Layer):

    def __init__(self, tensor_dims, **kwargs):
        """ General Linear Equivariant Layers

        Args:
            tensor_dims (int, int): calling with (2, 2) creates 2D tensor -> 2D tensor
            (2, 0) creates 2D tensor -> 0D tensor (perm invariant scalars) etc
        """
        super().__init__(**kwargs)

@keras.saving.register_keras_serializable(package='nano_pelican', name='Lineq2v2')
class Lineq2v2nano(Layer):
    """ This layer assumes the input 2D tensor is:
            1. Symmetric
            2. Hollow (zero diagonal)
        reducing the general 15 different linear equivariant transformations
        down to 6

    Args:
        Layer (_type_): _description_
    """
    def __init__(self, num_output_channels, activation=None, dropout=0.0, batchnorm=False, num_average_particles=1, **kwargs):
        super().__init__(**kwargs)

        self.output_channels = num_output_channels
        self.activation = activations.get(activation)

        self.dropout_rate = dropout
        self.use_batchnorm = batchnorm

        if self.dropout_rate > 0:
            self.dropout = Dropout(self.dropout_rate)

        if self.use_batchnorm:
            self.bnorm = BatchNormalization()

        self.average_particles = num_average_particles


    def build(self, input_shape):
        # input_shape = Batch x N x N x L Where L is the number
        # of output channels from the previous layer
        B, N, N, L = input_shape

        w_init = tf.random_normal_initializer(stddev=1/tf.cast(N, dtype=tf.float32))

        # For each input channel L, make 6 permequiv transformations,
        # and mix with a L*6 * self.output_channels dense layer

        self.w = self.add_weight(
                shape=(L, 6, self.output_channels),
                initializer=w_init,
                trainable=True,
        )

        b_init = tf.zeros_initializer()
        # Bias to be broadcasted over diagonal (for each channel)

        self.diag_bias = self.add_weight(
                shape=(self.output_channels, ),
                initializer=b_init,
                trainable=True,
        )

        # Bias to be broadcasted over entire output matrix (for each channel)
        self.bias = self.add_weight(
                shape=(self.output_channels, ),
                initializer=b_init,
                trainable=True,
        )


        # a_init = tf.random_uniform_initializer(minval=0, maxval=1)
        # self.alphas = self.add_weight(
        #         shape=(self.output_channels, ),
        #         initializer=a_init,
        #         trainable=True,
        # )

        super(Lineq2v2nano, self).build(input_shape)

    def call(self, inputs, training=False, *args, **kwargs):
        """
        input_shape : batch x N x N x L where L is number of input channels
        output_shape: batch x N x N x self.output_channels
        """

        if self.dropout_rate > 0:
            inputs = self.dropout(inputs, training=training)

        if self.use_batchnorm:
            inputs = self.bnorm(inputs, training=training)


        # B, N, N, L = inputs.shape
        N = inputs.shape[-2]

        totsum = tf.einsum("bijl->bl", inputs)/self.average_particles**2    # B x L
        rowsum = tf.einsum("biil->bil", inputs)/self.average_particles          # B x N x L

        ops = [None] * 6

        ONES = tf.ones((N, N), dtype=tf.float32)
        IDENTITY = tf.eye(N, dtype=tf.float32)

        # Identity
        ops[0] = inputs

        # Totsum over entire output
        ops[1] = tf.einsum("bl, ij->bijl", totsum, ONES)

        # Rowsum broadcasted over rows
        ops[2] = tf.einsum("bnl, nj->bnjl", rowsum, ONES)

        # Rowsum broadcasted over columns
        ops[3] = tf.einsum("bnl, in->binl", rowsum, ONES)

        # Rowsum broadcast over diagonals
        ops[4] = tf.einsum("bnl, nj->bnjl", rowsum, IDENTITY)

        #   totsum broadcast over diagonals
        ops[5] = tf.einsum("bl, ij->bijl", totsum, IDENTITY)
        ops = tf.stack(ops, axis=-1) # B x N x N x L x 6


        diag_bias = tf.einsum("f, ij->ijf", self.diag_bias, IDENTITY)
        return self.activation(
            (tf.einsum("bijlk, lkf->bijf", ops, self.w)
            + self.bias
            + diag_bias))
        #  * tf.pow(N/self.average_particles, self.alphas))

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                'num_output_channels': self.output_channels,
                'activation': self.activation,
                'dropout': self.dropout_rate,
                'batchnorm': self.use_batchnorm,
                'num_average_particles': self.average_particles
            }
        )

        return config

@keras.saving.register_keras_serializable(package='nano_pelican', name='Lineq2v0')
class Lineq2v0nano(Layer):
    def __init__(self, num_outputs, activation=None, dropout=0.0, batchnorm=False,num_average_particles=1, **kwargs):
        super().__init__(**kwargs)
        self.num_output_channels = num_outputs
        self.activation = activations.get(activation)

        self.dropout_rate = dropout
        self.use_batchnorm = batchnorm

        if self.dropout_rate > 0:
            self.dropout = Dropout(self.dropout_rate)

        if self.use_batchnorm:
            self.bnorm = BatchNormalization()

        self.average_particles = num_average_particles

    def build(self, input_shape):
        B, N, N, L = input_shape

        w_init = tf.random_normal_initializer(stddev=1/tf.cast(N, dtype=tf.float32))
        # For each input channel L, make 2 permequiv invariant,

        # and mix with a L*2 * self.num_output_channels dense layer
        self.w = self.add_weight(
                shape=(L, 2, self.num_output_channels),
                initializer=w_init,
                trainable=True,
        )

        b_init = tf.zeros_initializer()
        # Bias to be broadcasted over entire output

        self.bias = self.add_weight(
                shape=(self.num_output_channels, ),
                initializer=b_init,
                trainable=True,
        )

        super(Lineq2v0nano, self).build(input_shape)

    def call(self, inputs, training=False, *args, **kwargs):

        # inputs.shape = B x N x N x L

        if self.dropout_rate > 0:
            inputs = self.dropout(inputs, training=training)

        if self.use_batchnorm:
            inputs = self.bnorm(inputs, training=training)

        N = inputs.shape[-2]

        totsum  = tf.einsum("bijl->bl", inputs)/self.average_particles**2         # B x L
        trace   = tf.einsum("biil->bl", inputs)/self.average_particles      # B x L
        out     = tf.stack([totsum, trace], axis=-1)    # B x L x 2

        return self.activation(
            tf.einsum("blk, lkf->bf", out, self.w) + self.bias
        )

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                'num_outputs': self.num_output_channels,
                'activation': self.activation,
                'dropout': self.dropout_rate,
                'batchnorm': self.use_batchnorm,
                'num_average_particles': self.average_particles
            }
        )

        return config




