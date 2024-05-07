# The different linear equivariant layers
import tensorflow as tf
import logging

class LinearEquivariant(tf.keras.layers.Layer):
    def __init__(self, tensor_dims, **kwargs):
        """ General Linear Equivariant Layers

        Args:
            tensor_dims (int, int): calling with (2, 2) creates 2D tensor -> 2D tensor
            (2, 0) creates 2D tensor -> 0D tensor (perm invariant scalars) etc
        """
        super().__init__(**kwargs)

@tf.keras.utils.register_keras_serializable(package='nano_pelican', name='Lineq2v2')
class Lineq2v2nano(tf.keras.layers.Layer):
    """ This layer assumes the input 2D tensor is:
            1. Symmetric
            2. Hollow (zero diagonal)
        reducing the general 15 different linear equivariant transformations
        down to 6

    Args:
        Layer (_type_): _description_
    """
    def __init__(self, arg_dict):
        super().__init__()

        logger = logging.getLogger('')

        self.arg_dict = arg_dict
        self.output_channels = arg_dict['n_hidden']
        self.activation = tf.keras.activations.get(arg_dict['activation'])
        self.dropout_rate = arg_dict['dropout']
        self.use_batchnorm = arg_dict['batchnorm']
        self.average_particles = arg_dict['num_particles_avg']

        if self.dropout_rate > 0:
            logger.info("Using dropout in 2v2")
            self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        if self.use_batchnorm:
            logger.info("Using bnorm in 2v2")
            self.bnorm = tf.keras.layers.BatchNormalization()


    def compute_output_shape(self, input_shape):
        # input_shape =  N x N x L
        # out = Batch x N x N x self.output_channels
        N = input_shape[1]
        return (None, N, N, self.output_channels)


    def build(self, input_shape):
        # input_shape = Batch x N x N x L Where L is the number
        # of output channels from the previous layer
        if self.use_batchnorm:
            self.bnorm.build(input_shape)

        N = input_shape[-2]
        L = input_shape[-1]

        w_init = tf.random_normal_initializer(stddev=1/tf.cast(N, dtype=tf.dtypes.float32))

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

        super(Lineq2v2nano, self).build(input_shape)

    def call(self, inputs, training=False):
        """
        input_shape : batch x N x N x L where L is number of input channels
        output_shape: batch x N x N x self.output_channels
        """

        if self.dropout_rate > 0:
            inputs = self.dropout(inputs, training=training)

        if self.use_batchnorm:
            inputs = self.bnorm(inputs, training=training)



        totsum = tf.einsum("...ijl->...l", inputs)/self.average_particles**2    # B x L
        rowsum = tf.reduce_sum(inputs, axis=-2)/self.average_particles          # B x N x L

        ops = [None] * 6

        # B, N, N, L = inputs.shape
        N = tf.shape(inputs)[-2]
        ONES = tf.ones((N, N), dtype=tf.dtypes.float32)
        IDENTITY = tf.eye(N, dtype=tf.dtypes.float32)

        # Identity
        ops[0] = inputs

        # Totsum over entire output
        ops[1] = tf.einsum("...l, ij->...ijl", totsum, ONES)

        # Rowsum broadcasted over rows
        ops[2] = tf.einsum("...nl, nj->...njl", rowsum, ONES)

        # Rowsum broadcasted over columns
        ops[3] = tf.einsum("...nl, nj->...jnl", rowsum, ONES)

        # Rowsum broadcast over diagonals
        ops[4] = tf.einsum("...nl, nj->...njl", rowsum, IDENTITY)

        #   totsum broadcast over diagonals
        ops[5] = tf.einsum("...l, ij->...ijl", totsum, IDENTITY)

        ops = tf.stack(ops, axis=-1) # B x N x N x L x 6


        diag_bias = tf.einsum("f, ij->ijf", self.diag_bias, IDENTITY)
        return self.activation(
            (tf.einsum("...ijlk, lkf->...ijf", ops, self.w)
            + self.bias
            + diag_bias))

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                'arg_dict': self.arg_dict,
            }
        )

        return config

@tf.keras.utils.register_keras_serializable(package='nano_pelican', name='Lineq2v0')
class Lineq2v0nano(tf.keras.layers.Layer):
    def __init__(self, arg_dict):
        super().__init__()

        self.arg_dict = arg_dict
        self.num_output_channels = arg_dict['n_outputs']
        self.activation = tf.keras.activations.get(arg_dict['activation'])
        self.dropout_rate = arg_dict['dropout']
        self.use_batchnorm = arg_dict['batchnorm']
        self.average_particles = arg_dict['num_particles_avg']


        logger = logging.getLogger('')
        if self.dropout_rate > 0:
            logger.info("Using dropout in 2v0")
            self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        if self.use_batchnorm:
            logger.info("Using bnorm in 2v0")
            self.bnorm = tf.keras.layers.BatchNormalization()


    def build(self, input_shape):
        # N, N, L = input_shape
        if self.use_batchnorm:
            self.bnorm.build(input_shape)

        L = input_shape[-1]
        N = input_shape[-2]

        w_init = tf.random_normal_initializer(stddev=1/tf.cast(N, dtype=tf.dtypes.float32))
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

    def call(self, inputs, training=False):

        # inputs.shape = B x N x N x L

        if self.dropout_rate > 0:
            inputs = self.dropout(inputs, training=training)

        if self.use_batchnorm:
            inputs = self.bnorm(inputs, training=training)

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
                'arg_dict': self.arg_dict,
            }
        )

        return config




