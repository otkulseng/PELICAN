import tensorflow as tf
from keras import layers


class ScalingLayer(layers.Layer):
    def __init__(self, num_avg=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_avg = num_avg
        assert(num_avg > 0.0)


    def build(self, input_shape):
        L = input_shape[-1]
        N = input_shape[-2]
        assert(tf.abs(tf.cast(N, tf.float32) - self.num_avg) > 1.0e-6)
        # Scales with N / self.num_avg, so should not be equal

        alpha_init = tf.random_uniform_initializer(
            minval=0.0,
            maxval=1.0
        )
        self.alphas = self.add_weight(
                shape=(L,),
                initializer=alpha_init,
                trainable=True,
        )

    def calc_flops(self, input_shape):
        N = input_shape[-2]
        L = input_shape[-1]
        flops = {}

        flops['exponent'] = 0
        flops['product'] = N * N * L


        return flops

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        # x = layers.Activation('relu')(x)
        # delta =  (self.betas)**2
        # return ((1+x)**delta - 1) / delta
        N = tf.cast(x.shape[-2], dtype=tf.float32)
        return x * tf.pow(N / self.num_avg, self.alphas)
