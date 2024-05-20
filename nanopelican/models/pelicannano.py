# Based on https://github.com/abogatskiy/PELICAN-nano/
from keras.models import Model
from keras.layers import Input

from nanopelican import layers
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='nano_pelican', name='PelicanNano')
class PelicanNano(Model):
    def __init__(self, arg_dict):
        super(PelicanNano, self).__init__()

        self.arg_dict = arg_dict

        self.input_layer = layers.InnerProduct(arg_dict['input'])
        self.agg_layer = layers.Lineq2v2nano(arg_dict['lineq2v2'])
        self.out_layer = layers.Lineq2v0nano(arg_dict['lineq2v0'])

    def call(self, inputs, training=False):
        # inputs shape batch x N x CUSTOM
        # where data_format is supposed to convert to inner products

        inputs = self.input_layer(inputs)   # batch x N x N
        inputs = self.agg_layer(inputs, training=training)     # batch x N x N x hidden
        inputs = self.out_layer(inputs, training=training)     # batch x N x N x outputs


        return inputs

    def summary(self, input_shape):
        x = Input(shape=input_shape)
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()

    def build(self, input_shape):
        self.input_layer.build(input_shape)
        input_shape = self.input_layer.compute_output_shape(input_shape)

        self.agg_layer.build(input_shape)
        input_shape = self.agg_layer.compute_output_shape(input_shape)

        self.out_layer.build(input_shape)
        input_shape = self.out_layer.compute_output_shape(input_shape)


        self.output_shape = input_shape
        self.built = True

    def get_flops(self, input_shape):
        # no batch dim. Assumes num_particles x num_features
        flops = {}

        flops['inner_product'] = self.input_layer.get_flops(input_shape)
        input_shape = self.input_layer.compute_output_shape(input_shape)


        flops['lineq2v2'] = self.agg_layer.get_flops(input_shape)
        input_shape = self.agg_layer.compute_output_shape(input_shape)

        flops['lineq2v0'] = self.out_layer.get_flops(input_shape)
        return flops



    def get_config(self):
        config = {}

        config.update(
            {
                'arg_dict': self.arg_dict
            }
        )

        return config

































