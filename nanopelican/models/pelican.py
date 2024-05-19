# Based on https://github.com/abogatskiy/PELICAN-nano/
from keras.models import Model
from keras.layers import Input

from nanopelican import layers
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='nano_pelican', name='Pelican')
class Pelican(Model):
    def __init__(self, arg_dict):
        super(Pelican, self).__init__()

        self.arg_dict = arg_dict

        self.input_layer = layers.Lineq1v2(arg_dict['input'])
        self.agg_layer = layers.Lineq2v2nano(arg_dict['lineq2v2'])
        self.out_layer = layers.Lineq2v0nano(arg_dict['lineq2v0'])

    def call(self, inputs, training=False):
        # inputs shape batch x N x CUSTOM
        # where data_format is supposed to convert to inner products

        inputs = self.input_layer(inputs)   # batch x N x N x L
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



    def get_config(self):
        config = {}

        config.update(
            {
                'arg_dict': self.arg_dict
            }
        )

        return config

































