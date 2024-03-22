# Based on https://github.com/abogatskiy/PELICAN-nano/
import keras
from keras import backend as K
from keras.models import Model
from nanopelican import data

from nanopelican import layers

from pathlib import Path
import os

@keras.saving.register_keras_serializable(package='nano_pelican', name='PELICANnano')
class PELICANnano(Model):
    def __init__(self, hidden=1, outputs=2, activation='relu', data_format='fourvec',  *args, **kwargs):
        super().__init__(*args, **kwargs)


        self.hidden = hidden
        self.outputs = outputs
        self.activation = activation
        self.dataformat = data_format
        self.args = args
        self.kwargs = kwargs

        self.input_layer = layers.DataHandler(data_format=data_format)
        self.agg_layer = layers.Lineq2v2nano(num_output_channels=hidden, activation=activation)
        self.out_layer = layers.Lineq2v0nano(num_outputs=outputs, activation=None)

    def call(self, inputs, training=None, mask=None):
        # inputs shape batch x N x CUSTOM
        # where data_format is supposed to convert to inner products

        inputs = self.input_layer(inputs)   # batch x N x N x 1
        inputs = self.agg_layer(inputs)     # batch x N x N x hidden
        inputs = self.out_layer(inputs)     # batch x N x N x outputs
        return inputs

    def save_all_to_dir(self, dirname):
        counter = 0
        root = 'experiments'

        rootdir = Path.cwd() / root
        if not rootdir.exists():
            os.mkdir(rootdir)

        while True:
            self.path = (rootdir / f'{dirname}-{counter}')
            if not self.path.exists():
                break

            counter += 1

        os.mkdir(self.path)



    def get_config(self):
        config = super().get_config()

        config.update(
            {
                'hidden': self.hidden,
                'outputs': self.outputs,
                'activation': self.activation,
                'dataformat': self.data_format,
                'args': self.args,
                'kwargs': self.kwargs
            }
        )

        return config



































