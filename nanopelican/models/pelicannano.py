from os import name

from sympy import N
from nanopelican.layers import *
from keras import layers, Model

def PelicanNano(shape, conf):
    """ This model/layer takes as input the inner products,
    so needs to be composed with i.e. InnerProduct layer.
    """
    x = x_in = layers.Input(shape)

    x = Lineq2v2(hollow=True, symmetric=True)(x)
    x = layers.Dense(conf['n_hidden'], activation=conf['activation'])(x)
    x = Lineq2v0()(x)
    x = layers.Dense(conf['n_outputs'])(x)

    return Model(inputs=x_in, outputs=x, name='Pelican-Nano')


def MultiChannelPelican(shape, conf):
    x_in = layers.Input(shape)
    outputs = []
    for n in range(conf['num_channels']):
        x = PelicanNano()