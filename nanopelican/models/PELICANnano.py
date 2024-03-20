# Based on https://github.com/abogatskiy/PELICAN-nano/
import keras
from keras import backend as K
from keras.models import Model
from nanopelican import data

from nanopelican import layers

class PELICANnano(Model):
    def __init__(self, hidden=1, outputs=2, activation='relu', data_format='fourvec', batch_size=128, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_layer = layers.DataHandler(data_format=data_format)
        self.agg_layer = layers.Lineq2v2nano(num_output_channels=hidden, activation=activation)
        self.out_layer = layers.Lineq2v0nano(num_outputs=outputs, activation=None)

    def call(self, inputs, training=None, mask=None):
        # inputs shape batch x N x CUSTOM
        # where data_format is supposed to convert to inner products
        inputs = self.keras_input(inputs)

        inputs = self.input_layer(inputs)   # batch x N x N x 1
        inputs = self.agg_layer(inputs)     # batch x N x N x hidden
        inputs = self.out_layer(inputs)     # batch x N x N x outputs
        return inputs
































