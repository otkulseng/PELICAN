import tensorflow as tf
from keras import activations
from keras import layers
from keras import models

import numpy as np

class LinEq2v2(layers.Layer):
    def __init__(self, outputs=10, activation=None):
        super().__init__()

        self.activation = activations.get(activation)

        self.w = self.add_weight(
            shape=(15, outputs),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        totsum = tf.sum(inputs)
        trace = tf.einsum("ii", inputs)
        diag = tf.einsum("ii->i", inputs)
        rowsum = tf.einsum("ij->i", inputs) # i'th element is sum of i'th row
        colsum = tf.einsum("ij->j", inputs) # i'th element is sum of i'th column

        output = [None] * 15
        output[0] = tf.linalg.diag(diag)
        output[1] = tf.linalg.diag(rowsum)
        output[2] = tf.linalg.diag(colsum)
        output[3] = tf.identity(diag.size) * trace
        output[4] = tf.identity(diag.sie) * totsum
        output[5] = tf.einsum("i, j->ij", tf.ones_like(diag), diag)
        output[6] = tf.einsum("i, j->ij", tf.ones_like(rowsum), rowsum)
        output[7] = tf.einsum("i, j->ij", tf.ones_like(colsum), colsum)
        output[8] = tf.einsum("i, j->ji", tf.ones_like(diag), diag)
        output[9] = tf.einsum("i, j->ji", tf.ones_like(rowsum), rowsum)
        output[10] = tf.einsum("i, j->ji", tf.ones_like(colsum), colsum)
        output[11] = tf.transpose(inputs)
        output[12] = inputs
        output[13] = tf.ones_like(inputs)*trace
        output[14] = tf.ones_like(inputs)*totsum

        output = tf.stack(output, axis=-1)
        return self.activation(tf.matmul(output, self.w))



class EquivariantBlock(layers.Layer):
    def __init__(self):
        super().__init__()

        self.msg_relu = layers.LeakyReLU()
        self.msg_batch = layers.BatchNormalization()
        self.msg_dropout = layers.Dropout(0.2)

        self.agg_lineq = LinEq2v2(activation='relu')



    def call(self, inputs):
        pass

