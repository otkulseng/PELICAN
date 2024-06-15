from keras import layers
import tensorflow as tf

class ApproxLog(layers.Layer):
    def build(self, input_shape):
        L = input_shape[-1]

        alpha_init = tf.ones_initializer()

        self.alphas = self.add_weight(
                shape=(2,),
                initializer=alpha_init,
                trainable=True,
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        x = layers.Activation('relu')(x)
        return self.alphas[0] * x / (1 + x/self.alphas[1]**2)