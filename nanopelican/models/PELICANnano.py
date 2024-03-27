# Based on https://github.com/abogatskiy/PELICAN-nano/
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Reshape
from nanopelican import data

from nanopelican import layers
import tensorflow as tf


from pathlib import Path
import os
import pickle


@keras.saving.register_keras_serializable(package='nano_pelican', name='PelicanNano')
class PelicanNano(Model):
    def __init__(self, cli_args,**kwargs):
        super(PelicanNano, self).__init__(**kwargs)

        self.cli_args = cli_args

        self.input_layer = layers.InnerProduct(
            data_format=cli_args['data_format'],
            num_particles=cli_args['num_particles'])
        self.agg_layer = layers.Lineq2v2nano(
            num_output_channels=cli_args['n_hidden'],
            activation=cli_args['activation'],
            dropout=cli_args['dropout_rate'],
            batchnorm=cli_args['use_batchnorm'],
            num_average_particles=cli_args['num_particles_avg'])
        self.out_layer = layers.Lineq2v0nano(
            num_outputs=cli_args['n_outputs'],
            activation=None,
            dropout=cli_args['dropout_rate'],
            batchnorm=cli_args['use_batchnorm'],
            num_average_particles=cli_args['num_particles_avg'])

    def build(self, input_shape):
        self.call(tf.zeros(input_shape))


    def call(self, inputs, training=False):
        # inputs shape batch x N x CUSTOM
        # where data_format is supposed to convert to inner products

        inputs = self.input_layer(inputs)   # batch x N x N x 1

        inputs = self.agg_layer(inputs, training=training)     # batch x N x N x hidden
        inputs = self.out_layer(inputs, training=training)     # batch x N x N x outputs
        return inputs

    def save_all_to_dir(self, args):
        counter = 0
        root = args.experiment_root

        rootdir = Path.cwd() / root
        if not rootdir.exists():
            os.mkdir(rootdir)

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
                'cli_args': self.cli_args
            }
        )

        return config

































