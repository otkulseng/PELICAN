# Based on https://github.com/abogatskiy/PELICAN-nano/
from keras import backend as K
from keras.models import Model
from keras.layers import Reshape, Input
from nanopelican import data

from nanopelican import layers
from nanopelican import data
import tensorflow as tf


from pathlib import Path
import os
import pickle


@tf.keras.utils.register_keras_serializable(package='nano_pelican', name='PelicanNano')
class PelicanNano(Model):
    def __init__(self, arg_dict):
        super(PelicanNano, self).__init__()

        self.arg_dict = arg_dict

        self.input_layer = layers.InnerProduct(arg_dict['input'])
        self.agg_layer = layers.Lineq2v2nano(arg_dict['lineq2v2'])
        self.out_layer = layers.Lineq2v0nano(arg_dict['lineq2v0'])

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
        self.built = True


    def call(self, inputs, training=False):
        # inputs shape batch x N x CUSTOM
        # where data_format is supposed to convert to inner products

        inputs = self.input_layer(inputs)   # batch x N x N
        inputs = self.agg_layer(inputs, training=training)     # batch x N x N x hidden
        inputs = self.out_layer(inputs, training=training)     # batch x N x N x outputs


        return inputs

    def save_all_to_dir(self, args):
        root = args.experiment_root

        rootdir = Path.cwd() / root
        if not rootdir.exists():
            os.mkdir(rootdir)

        counter = 0
        while True:
            self.folder = (rootdir / f'{args.experiment_name}-{counter}')
            if not self.folder.exists():
                break

            counter += 1

        os.mkdir(self.folder)

        super().save(self.folder / 'model.keras')
        with open(self.folder / 'history.pkl', 'wb') as file_pi:
            mydict = self.history.history
            mydict['args'] = args
            pickle.dump(mydict, file_pi)

        with open(self.folder / 'summary.txt', 'w') as summary_file:
            for key, val in args.__dict__.items():
                summary_file.write(f'{key} : {val} \n')

    def get_config(self):
        config = {}

        config.update(
            {
                'arg_dict': self.arg_dict
            }
        )

        return config

































