import tensorflow as tf
from keras import activations
from keras import layers

class LinEq2v2(layers.Layer):
    def __init__(self, activation=None):
        self.lambdas = self.add_weight(
            shape=(15, 1), # 15 as there are 15 different PermEq tensors between two 2d tensors
            initializer="random_normal",
            trainable=True,
        )

        self.activation = activations.get(activation)

    def call(self, inputs):
        totsum = tf.sum(inputs)
        trace = tf.einsum("ii", inputs)
        diag = tf.einsum("ii->i", inputs)
        rowsum = tf.einsum("ij->i", inputs) # i'th element is sum of i'th row
        colsum = tf.einsum("ij->j", inputs) # i'th element is sum of i'th column

        # To diagonal (1, 4, 5, 6, 12)
        b = tf.diag(
            self.lambdas[0] * diag +
            self.lambdas[1] * rowsum +
            self.lambdas[2] * colsum +
            self.lambdas[3] * trace +
            self.lambdas[4] * totsum
        )

        # To cols (2, 13, 14).
        # Note: Cols, as transpose later.
        # Transpose does not affect diag.
        b += (
            self.lambdas[5] * diag +
            self.lambdas[6] * rowsum +
            self.lambdas[7] * colsum
        )

        # To cols (3, 10, 11)
        b = tf.transpose(b)
        b += (
            self.lambdas[8] * diag +
            self.lambdas[9] * rowsum +
            self.lambdas[10] * colsum
        )
        # b = tf.transpose(b)

        # To whole (7, 8, 9, 15)
        b += (
            self.lambdas[11] * tf.transpose(inputs) +
            self.lambdas[12] * inputs +
            self.lambdas[13] * trace +
            self.lambdas[14] * totsum
        )
        return self.activation(b)




