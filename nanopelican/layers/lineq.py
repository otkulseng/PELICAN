import tensorflow as tf
from keras import layers

class Lineq2v0(layers.Layer):
    def __init__(self, hollow=False, num_avg=1.0,*args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_totsum = True
        self.use_trace = not hollow
        self.num_avg = num_avg

    def call(self, inputs, training=False):
        ops = []

        if self.use_totsum:
            ops.append(tf.einsum("...ijl->...l", inputs) / self.num_avg**2)    # B x L

        if self.use_trace:
            ops.append(tf.einsum("...iil->...l", inputs) / self.num_avg)     # B x L

        return tf.concat(ops, axis=-1)   # B x L x 2



class Lineq2v2(layers.Layer):
    """ Aggregates the 2D input tensor into the (max) 15 different permutation
    equivariant tensors that can be made.

    Set symmetric=True if the input tensor is symmetric
    Set hollow=True if the input tensor has zero trace

    """
    def __init__(self, symmetric=False, hollow=False, num_avg=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.symmetric = symmetric
        self.hollow = hollow
        self.num_avg = num_avg

        self.use_totsum     = True
        self.use_rowsum     = True
        self.use_colsum     = True
        self.use_trace      = True
        self.use_diag       = True
        self.use_identity   = True
        self.use_transpose  = True

        if self.symmetric:
            self.use_colsum = False
            self.use_transpose = False

        if self.hollow:
            self.use_colsum = False
            self.use_trace  = False
            self.use_diag   = False



    def call(self, inputs, training=False):
        # B, N, N, L = inputs.shape
        N = tf.shape(inputs)[-2]

        ops = []
        ONES = tf.ones((N, N), dtype=tf.dtypes.float32)
        IDENTITY = tf.eye(N, dtype=tf.dtypes.float32)

        if self.use_totsum:
            totsum  = tf.einsum('...ijl->...l', inputs) / self.num_avg**2
            # broadcast over diagonals
            ops.append(tf.einsum("...l, ij->...ijl", totsum, IDENTITY))

            # broadcast over entire output
            ops.append(tf.einsum("...l, ij->...ijl", totsum, ONES))

        if self.use_trace:
            trace   = tf.einsum('...iil->...l', inputs) / self.num_avg

            # broadcast over diagonals
            ops.append(tf.einsum("...l, ij->...ijl", trace, IDENTITY))

            # broadcast over entire output
            ops.append(tf.einsum("...l, ij->...ijl", trace, ONES))

        if self.use_rowsum:
            rowsum  = tf.einsum('...ijl->...il', inputs) / self.num_avg
            # broadcasted over rows
            ops.append(tf.einsum("...nl, nj->...njl", rowsum, ONES))

            # broadcasted over columns
            ops.append(tf.einsum("...nl, nj->...jnl", rowsum, ONES))

            # broadcast over diagonals
            ops.append(tf.einsum("...nl, nj->...njl", rowsum, IDENTITY))


        if self.use_colsum:
            colsum  = tf.einsum('...ijl->...jl', inputs) / self.num_avg
            # broadcasted over rows
            ops.append(tf.einsum("...nl, nj->...njl", colsum, ONES))

            # broadcasted over columns
            ops.append(tf.einsum("...nl, nj->...jnl", colsum, ONES))

            # broadcast over diagonals
            ops.append(tf.einsum("...nl, nj->...njl", colsum, IDENTITY))


        if self.use_diag:
            diag    = tf.einsum('...iil->...il', inputs)
            # broadcasted over rows
            ops.append(tf.einsum("...nl, nj->...njl", diag, ONES))

            # broadcasted over columns
            ops.append(tf.einsum("...nl, nj->...jnl", diag, ONES))

            # broadcast over diagonals
            ops.append(tf.einsum("...nl, nj->...njl", diag, IDENTITY))


        if self.use_identity:
            ops.append(inputs)

        if self.use_transpose:
            ops.append(tf.einsum("...ijl->...jil", inputs))


        return tf.concat(ops, axis=-1)


class Lineq1v2(layers.Layer):
    """ Aggregates the 2D input tensor into the (max) 15 different permutation
    equivariant tensors that can be made.

    Set symmetric=True if the input tensor is symmetric
    Set hollow=True if the input tensor has zero trace

    """
    def __init__(self, hollow=False, num_avg=1.0,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_avg = num_avg

        self.hollow = hollow
        self.use_totsum = True
        if self.hollow:
            self.use_totsum = False

    def call(self, inputs, training=False):
        # B, N, L = inputs.shape
        N = tf.shape(inputs)[-2]

        ops = []
        ONES = tf.ones((N, N), dtype=tf.dtypes.float32)
        IDENTITY = tf.eye(N, dtype=tf.dtypes.float32)


        # broadcasted over rows
        ops.append(tf.einsum("...nl, nj->...njl", inputs, ONES))

        # broadcasted over columns
        ops.append(tf.einsum("...nl, nj->...jnl", inputs, ONES))

        # broadcast over diagonals
        ops.append(tf.einsum("...nl, nj->...njl", inputs, IDENTITY))

        if self.use_totsum:
            totsum = tf.einsum('...il->...l', inputs)
            # broadcast over diagonals
            ops.append(tf.einsum("...l, ij->...ijl", totsum, IDENTITY))

            # broadcast over entire output
            ops.append(tf.einsum("...l, ij->...ijl", totsum, ONES))


        return tf.concat(ops, axis=-1)