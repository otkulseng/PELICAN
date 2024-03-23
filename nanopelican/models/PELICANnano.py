# Based on https://github.com/abogatskiy/PELICAN-nano/
import keras
from keras import backend as K
from keras.models import Model
from nanopelican import data

from nanopelican import layers

from pathlib import Path
import os
import pickle
from keras.models import load_model
from keras.callbacks import History

@keras.saving.register_keras_serializable(package='nano_pelican', name='PELICANnano')
class PELICANnano(Model):
    def __init__(self, hidden=1, outputs=2, activation='relu', data_format='fourvec'):
        super().__init__()

        self.hidden = hidden
        self.outputs = outputs
        self.activation = activation
        self.dataformat = data_format

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

    def save_all_to_dir(self, dirname, args={}):
        counter = 0
        root = 'experiments'

        rootdir = Path.cwd() / root
        if not rootdir.exists():
            os.mkdir(rootdir)

        while True:
            self.folder = (rootdir / f'{dirname}-{counter}')
            if not self.folder.exists():
                break

            counter += 1

        os.mkdir(self.folder)

        super().save(self.folder / 'model.keras')
        with open(self.folder / 'history.pkl', 'wb') as file_pi:
            mydict = self.history.history
            mydict['args'] = args
            pickle.dump(mydict, file_pi)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                'hidden': self.hidden,
                'outputs': self.outputs,
                'activation': self.activation,
                'data_format': self.dataformat
            }
        )

        return config

































