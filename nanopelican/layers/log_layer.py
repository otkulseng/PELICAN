from keras import layers
import tensorflow as tf

class LogLayer(layers.Layer):
    def build(self, input_shape):
        L = input_shape[-1]

        alpha_init = tf.random_uniform_initializer()

        self.betas = self.add_weight(
                shape=(L,),
                initializer=alpha_init,
                trainable=True,
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        x = layers.Activation('relu')(x)
        delta =  (self.betas)**2
        return ((1+x)**delta - 1) / delta