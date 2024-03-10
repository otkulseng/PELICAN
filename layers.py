import tensorflow as tf
from keras import activations
from keras import layers
from keras import models

# Message forming layer
class Msg(layers.Layer):
    def __init__(self, outputs, activation='leaky_relu'):
        super().__init__()

        self.activation = activations.get(activation)
        self.num_outputs = outputs

    def build(self, input_shape):
        # Assumes input_shape is a list (tensor) of permutation
        # equivariant 2d tensors indexed by last index.
        # i.e. input_shape = (batch) x N x N x L where
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
        x = tf.einsum("...ijl, lf->...ijf", inputs, self.w)
        x = self.activation(x)
        return self.bnorm(x, training=training)


# Aggregation layers
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
        # Tests for all these in equivariant.py
        totsum = tf.einsum("...ijl -> ...l", inputs)
        trace = tf.einsum("...iil->...l", inputs)
        diag = tf.einsum("...iil ->...li", inputs)
        rowsum = tf.einsum("...ijl -> ...li", inputs)
        colsum = tf.einsum("...ijl -> ...lj", inputs)


        output = [None] * 15

        # The diagonal, rowsum and colsum broadcasted over the diagonal
        output[0] = tf.einsum("...lij->...ijl", tf.linalg.diag(diag)) #diag_to_diag_1
        output[1] = tf.einsum("...lij->...ijl", tf.linalg.diag(rowsum)) #rowsum_to_diag_4
        output[2] = tf.einsum("...lij->...ijl", tf.linalg.diag(colsum)) #colsum_to_diag_5

        # The trace and total sum of the matrices broadcasted over diagonal
        shape = trace.shape.as_list()
        if shape[0] == None:
            shape[0] = 1
        A = tf.eye(num_rows=diag.shape[-1], batch_shape=shape) # batch x L x (eye(N))
        output[3] = tf.einsum("...l, ...lij->...ijl", trace, A) #trace_to_diag_9
        output[4] = tf.einsum("...l, ...lij->...ijl", totsum, A) #totsum_to_diag_12

        # The diagonal, rowsum and colsum broadcasted over the rows
        output[5] = tf.einsum("...li, ...lj ->...ijl", tf.ones_like(diag), diag) #diag_to_rows_2
        output[7] = tf.einsum("...li, ...lj->...ijl", tf.ones_like(colsum), colsum) #colsum_to_rows_13
        output[6] = tf.einsum("...li, ...lj->...ijl", tf.ones_like(rowsum), rowsum) #rowsum_to_rows_14

        # The diagonal, rowsum and colsum broadcasted over the columns
        output[8] = tf.einsum("...li, ...lj->...jil", tf.ones_like(diag), diag) #diag_to_cols_3
        output[9] = tf.einsum("...li, ...lj->...jil", tf.ones_like(rowsum), rowsum) #rowsum_to_cols_10
        output[10] = tf.einsum("...li, ...lj->...jil", tf.ones_like(colsum), colsum) #colsum_to_cols_11

        # Identity, transpose, trace and totsum broadcasted over entire output
        output[11] = tf.einsum("...ijl->...jil", inputs) #transpose_to_all_7
        output[12] = inputs
        output[13] = tf.einsum("...ijl, ...l->...ijl", tf.ones_like(inputs), trace) #trace_to_all_6
        output[14] = tf.einsum("...ijl, ...l->...ijl", tf.ones_like(inputs), totsum) #totsum_to_all_15

        output = tf.stack(output, axis=-1)
        # output is now B x N x N x L x 15 tensor. Now, all L x 15 tensors are
        # permutation invariant. Any linear combination of these are therefore
        # also permutation invariant. The weights of this layer has shape L x 15 x self.num_outputs.
        # The calculation below thus yield a B x N x N x self.num_outputs tensor.


        # See https://proceedings.mlr.press/v151/pan22a/pan22a.pdf#page=13
        return self.activation(tf.einsum("...de,def->...f", output, self.w))

class LinEq2v0(layers.Layer):
    def __init__(self, outputs, activation=None):
        super().__init__()

        self.activation = activations.get(activation)
        self.num_outputs = outputs

    def build(self, input_shape):
        # Note: See comment under MSG layer build
        l = input_shape[-1]

        self.w = self.add_weight(
            shape=(l, 2, self.num_outputs),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        totsum = tf.einsum("...ijl -> ...l", inputs)
        trace = tf.einsum("...iil->...l", inputs)

        output = tf.stack([totsum, trace], axis=-1)

        return self.activation(tf.einsum("...de,def->...f", output, self.w))


