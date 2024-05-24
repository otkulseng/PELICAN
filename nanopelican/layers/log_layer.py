from keras.layers import Layer
import tensorflow as tf

class LogLayer(Layer):


    def build(self, input_shape):
        L = input_shape[-1]

        alpha_init = tf.random_normal_initializer(stddev=1/tf.cast(L, dtype=tf.dtypes.float32))

        self.alphas = self.add_weight(
                shape=(L,),
                initializer=alpha_init,
                trainable=True,
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        return tf.math.pow(1+x, self.alphas)/(1e-6 + self.alphas**2)