# The different linear equivariant layers
import tensorflow as tf
from keras.layers import Layer
from keras import activations
from keras import backend as K

class LinearEquivariant(Layer):

    def __init__(self, tensor_dims, **kwargs):
        """ General Linear Equivariant Layers

        Args:
            tensor_dims (int, int): calling with (2, 2) creates 2D tensor -> 2D tensor
            (2, 0) creates 2D tensor -> 0D tensor (perm invariant scalars) etc
        """
        super().__init__(**kwargs)

class Lineq2v2nano(Layer):
    """ This layer assumes the input 2D tensor is:
            1. Symmetric
            2. Hollow (zero diagonal)
        reducing the general 15 different linear equivariant transformations
        down to 6

    Args:
        Layer (_type_): _description_
    """
    def __init__(self, num_output_channels, activation=None, **kwargs):
        super().__init__(**kwargs)

        self.output_channels = num_output_channels
        self.activation = activations.get(activation)

    def build(self, input_shape):
        # input_shape = Batch x N x N x L Where L is the number
        # of output channels from the previous layer
        B, N, N, L = input_shape

        w_init = tf.random_normal_initializer(stddev=1/N)

        # For each input channel L, make 6 permequiv transformations,
        # and mix with a L*6 * self.output_channels dense layer
        self.w = tf.Variable(
            initial_value=w_init(
                shape=(L, 6, self.output_channels),
                dtype='float32'
            ),
            trainable=True
        )

        b_init = tf.zeros_initializer()
        # Bias to be broadcasted over diagonal (for each channel)
        self.diag_bias = tf.Variable(
            initial_value=b_init(
                shape=(self.output_channels, ),
                dtype='float32'
            ),
            trainable=True
        )

        # Bias to be broadcasted over entire output matrix (for each channel)
        self.bias = tf.Variable(
            initial_value=b_init(
                shape=(self.output_channels, ),
                dtype='float32'
            ),
            trainable=True
        )

        super(Lineq2v2nano, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        """
        input_shape : batch x N x N x L where L is number of input channels
        output_shape: batch x N x N x self.output_channels
        """

        print(self.w)

        B, N, N, L = inputs.shape

        totsum = K.sum(inputs, axis=(1, 2))  # B x L
        rowsum = K.sum(inputs, axis=1)       # B x N x L

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
        diag_bias = tf.einsum("f, ij->ijf", self.diag_bias, IDENTITY)

        ops = tf.stack(ops, axis=-1) # B x N x N x L x 6

        return self.activation(
            tf.einsum("bijlk, lkf->bijf", ops, self.w)
            + self.bias
            + diag_bias
        )

class Lineq2v0nano(Layer):
    def __init__(self, num_outputs, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.num_outputs = num_outputs
        self.activation = activations.get(activation)

    def build(self, input_shape):
        B, N, N, L = input_shape

        w_init = tf.random_normal_initializer(stddev=1/N)
        # For each input channel L, make 2 permequiv invariant,
        # and mix with a L*2 * self.num_outputs dense layer
        self.w = tf.Variable(
            initial_value=w_init(
                shape=(L, 2, self.num_outputs),
                dtype='float32'
            ),
            trainable=True
        )

        b_init = tf.zeros_initializer()
        # Bias to be broadcasted over entire output
        self.bias = tf.Variable(
            initial_value=b_init(
                shape=(self.num_outputs, ),
                dtype='float32'
            ),
            trainable=True
        )

        super(Lineq2v0nano, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        # inputs.shape = B x N x N x L
        totsum  = tf.einsum("bijl->bl", inputs)         # B x L
        trace   = tf.einsum("biil->bl", inputs)         # B x L
        out     = tf.stack([totsum, trace], axis=-1)    # B x L x 2

        return self.activation(
            tf.einsum("blk, lkf->bf", out, self.w) + self.bias
        )




