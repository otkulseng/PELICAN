import tensorflow as tf
from keras import activations
from keras import layers
from keras import models

import numpy as np
# Message forming layer
class Msg(layers.Layer):
    def __init__(self, outputs, activation='leaky_relu'):
        super().__init__()

        self.activation = activations.get(activation)
        self.num_outputs = outputs

    def build(self, input_shape):
        # Assumes input_shape is a list (tensor) of permutation
        # equivariant 2d tensors indexed by last index.
        # i.e. input_shape = batch x N x N x L where
        # N is dimension of tensor and L is length of the list.
        # Any linear combination of perm-eq tensors are perm-eq
        l = input_shape[-1]

        self.w = self.add_weight(
            shape=(l, self.num_outputs),
            initializer="random_normal",
            trainable=True,
        )

        self.bnorm = layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = tf.matmul(inputs, self.w)
        x = self.activation(x)
        return self.bnorm(x, training=training)


# Aggregation layer
class LinEq2v2(layers.Layer):
    def __init__(self, outputs, activation=None):
        super().__init__()

        self.activation = activations.get(activation)
        self.num_outputs = outputs

    def build(self, input_shape):
        # Note: See comment under MSG layer build
        l = input_shape[-1]

        self.w = self.add_weight(
            shape=(l, 15, self.num_outputs),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        # Shape of inputs is
        # batch x N x N x L
        N = inputs[-2]

        totsum = tf.einsum("...ijl -> ...l", inputs)
        trace = tf.einsum("...iil->...l", inputs)
        diag = tf.einsum("...iil ->...li", inputs) # diag i indexed by l
        rowsum = tf.einsum("...ijl -> ...li", inputs)
        colsum = tf.einsum("...ijl -> ...lj", inputs)

        output = [None] * 15

        # The diagonal, rowsum and colsum broadcasted over the diagonal
        output[0] = tf.linalg.diag(diag)
        output[1] = tf.linalg.diag(rowsum)
        output[2] = tf.linalg.diag(colsum)

        # The trace and total sum of the matrices broadcasted over diabonal
        A = tf.eye(num_rows=N, batch_shape=trace.shape) # batch x L x (eye(N))
        output[3] = np.einsum("...i, ...ijk->...ijk", trace, A)
        output[4] = np.einsum("...i, ...ijk->...ijk", totsum, A)

        # The diagonal, rowsum and colsum broadcasted over the rows
        output[5] = tf.einsum("...i, ...j ->...ij", tf.ones_like(diag), diag)
        output[6] = tf.einsum("...i, ...j->...ij", tf.ones_like(rowsum), rowsum)
        output[7] = tf.einsum("...i, ...j->...ij", tf.ones_like(colsum), colsum)

        # The diagonal, rowsum and colsum broadcasted over the columns
        output[8] = tf.einsum("...i, ...j->...ji", tf.ones_like(diag), diag)
        output[9] = tf.einsum("...i, ...j->...ji", tf.ones_like(rowsum), rowsum)
        output[10] = tf.einsum("...i, ...j->...ji", tf.ones_like(colsum), colsum)

        # Identity, transpose, trace and totsum broadcasted over entire output
        output[11] = tf.einsum("..ij->ji", inputs)
        output[12] = inputs
        output[13] = tf.ones_like(inputs)*trace
        output[14] = tf.ones_like(inputs)*totsum

        output = tf.stack(output, axis=-1)
        # output is now B x N x N x L x 15 tensor. Now, all L x 15 tensors are
        # permutation invariant. Any linear combination of these are therefore
        # also permutation invariant. The weights of this layer has shape L x 15 x self.num_outputs.
        # The calculation below thus yield a B x N x N x self.num_outputs tensor.

        return self.activation(tf.einsum("...de,def->...f", output, self.w))

        # totsum = tf.sum(inputs)
        # trace = tf.einsum("ii", inputs)
        # diag = tf.einsum("ii->i", inputs)
        # rowsum = tf.einsum("ij->i", inputs) # i'th element is sum of i'th row
        # colsum = tf.einsum("ij->j", inputs) # i'th element is sum of i'th column


        # output = [None] * 15
        # output[0] = tf.linalg.diag(diag)
        # output[1] = tf.linalg.diag(rowsum)
        # output[2] = tf.linalg.diag(colsum)

        # # L = np.einsum("...iil->...l", b) : list of traces by L. Or batch x L
        # A = tf.eye(num_rows=N, batch_shape=trace.shape)
        # B = np.einsum("...i, ...ijk->...ijk", trace, A)

        # output[3] = tf.identity(diag.size) * trace
        # output[4] = tf.identity(diag.sie) * totsum
        # output[5] = tf.einsum("i, j->ij", tf.ones_like(diag), diag)
        # output[6] = tf.einsum("i, j->ij", tf.ones_like(rowsum), rowsum)
        # output[7] = tf.einsum("i, j->ij", tf.ones_like(colsum), colsum)
        # output[8] = tf.einsum("i, j->ji", tf.ones_like(diag), diag)
        # output[9] = tf.einsum("i, j->ji", tf.ones_like(rowsum), rowsum)
        # output[10] = tf.einsum("i, j->ji", tf.ones_like(colsum), colsum)
        # output[11] = tf.transpose(inputs)
        # output[12] = inputs
        # output[13] = tf.ones_like(inputs)*trace
        # output[14] = tf.ones_like(inputs)*totsum

        # output = tf.stack(output, axis=-1)


class EquivariantBlock(layers.Layer):
    def __init__(self, depth=1, dropout=0, msg_outputs=10, agg_outputs=10, activation=None):
        super().__init__()

        self.msg_layers = [Msg(outputs=msg_outputs, activation=activation) for _ in range(depth)]
        self.dropout = layers.Dropout(rate=dropout)
        self.agg_layers = [LinEq2v2(outputs=agg_outputs, activation=activation) for _ in range(depth)]

    def call(self, inputs):
        x = inputs
        for msg, agg in zip(self.msg_layers, self.agg_layers):
            x = msg(inputs)
            x = self.dropout(x)
            x = agg(x)
        return x

