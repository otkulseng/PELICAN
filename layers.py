import tensorflow as tf
from keras import activations
from keras import layers
from keras import models

# Message forming layer
class InputLayer(layers.Layer):

    def build(self, input_shape):
        N = input_shape[-2]
        L = input_shape[-1]

        self.w = self.add_weight(
            shape=(L),
            initializer="random_normal",
            trainable=True,
        )


        return super().build(input_shape)

    def call(self, inputs, training=False):
        N = inputs.shape[-2]
        L = inputs.shape[-1]
        inputs = tf.reshape(inputs, shape=(-1, N, N, L))



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
        x = tf.einsum("bijl, lf->bijf", inputs, self.w)
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

        self.n = input_shape[-2]

        self.w = self.add_weight(
            shape=(15, l, self.num_outputs),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):

        batch, N, _, L = inputs.shape
        batch = 1 if batch is None else batch

        totsum  = tf.einsum("bijl -> bl ", inputs)
        trace   = tf.einsum("biil -> bl ", inputs)
        diag    = tf.einsum("biil -> bli", inputs)
        rowsum  = tf.einsum("bijl -> bli", inputs)
        colsum  = tf.einsum("bijl -> blj", inputs)


        # diagonal (bli), eye(N) (ij), self.weight[0] (lf)
        res0 = tf.einsum("bli, ij, lf->bijf", diag, tf.eye(N), self.w[0])
        res1 = tf.einsum("bli, ij, lf->bijf", rowsum, tf.eye(N), self.w[1])
        res2 = tf.einsum("bli, ij, lf->bijf", colsum, tf.eye(N), self.w[2])

        # trace (bl) self.w[] (lf) og eye(N) (ij)
        res3 = tf.einsum("bl, ij, lf->bijf", trace, tf.eye(N), self.w[3])
        res4 = tf.einsum("bl, ij, lf->bijf", totsum, tf.eye(N), self.w[4])

        # diag (bli) self.w (lf) ones(N) (ij)
        res5 = tf.einsum("blj, ij, lf->bijf", diag, tf.ones((N, N)), self.w[5])
        res6 = tf.einsum("blj, ij, lf->bijf", rowsum, tf.ones((N, N)), self.w[6])
        res7 = tf.einsum("blj, ij, lf->bijf", colsum, tf.ones((N, N)), self.w[7])

        # diag (bli) self.w (lf) ones(N) (ij)
        res8 = tf.einsum("bli, ij, lf->bijf", diag, tf.ones((N, N)), self.w[8])
        res9 = tf.einsum("bli, ij, lf->bijf", rowsum, tf.ones((N, N)), self.w[9])
        res10 = tf.einsum("bli, ij, lf->bijf", colsum, tf.ones((N, N)), self.w[10])


        res11 = tf.einsum("bijl, lf->bjif", inputs, self.w[11]) #transpose
        res12 = tf.einsum("bijl, lf->bijf", inputs, self.w[12])

        # trace (bl) self.w[] (lf) og eye(N) (ij)
        res13 = tf.einsum("bl, ij, lf->bijf", trace, tf.ones((N, N)), self.w[13])

        # totsum (bl) self.w[] (lf) og eye(N) (ij)
        res14 = tf.einsum("bl, ij, lf->bijf", totsum, tf.ones((N, N)), self.w[14])

        # 128 x 100 x 100 x 15 floats 60x35


        return tf.add_n(
            [res0,
             res1,
             res2,
             res3,
             res4,
             res5,
             res6,
             res7,
             res8,
             res9,
             res10,
             res11,
             res12,
             res13,
             res14
             ]
        )


    def call_old(self, inputs):
        # Tests for all these in equivariant.py

        batch, N, _, L = inputs.shape
        batch = 1 if batch is None else batch

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
        A = tf.eye(num_rows=N, batch_shape=(batch, L)) # batch x L x (eye(N))
        output[3] = tf.einsum("...l, ...lij->...ijl", trace, A) #trace_to_diag_9
        output[4] = tf.einsum("...l, ...lij->...ijl", totsum, A) #totsum_to_diag_12

        # The diagonal, rowsum and colsum broadcasted over the rows
        output[5] = tf.einsum("...li, ...lj ->...ijl", tf.ones_like(diag), diag) #diag_to_rows_2
        output[6] = tf.einsum("...li, ...lj->...ijl", tf.ones_like(rowsum), rowsum) #rowsum_to_rows_14
        output[7] = tf.einsum("...li, ...lj->...ijl", tf.ones_like(colsum), colsum) #colsum_to_rows_13

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


