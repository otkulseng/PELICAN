
from keras import layers
from nanopelican.layers import *
import tensorflow as tf
import functools
from collections.abc import Iterable

def input_layer(input_shape, layer):
    return input_shape, {}


def pelican_layer(input_shape, layer):
    out_shape = layer.call(tf.zeros(input_shape)).shape
    return out_shape, layer.calc_flops(input_shape)


def activation(input_shape, layer):
    out_shape = layer.call(tf.ones(input_shape)).shape
    return out_shape, {
        'activation' : get_flops_activ(input_shape, layer.activation.__name__)
    }

def dense(input_shape, layer):
    out_shape = layer.call(tf.ones(input_shape)).shape

    # input_shape N x N x n_in
    # output_shape N x N x n_out
    # can be seen as a regular dense network for all N x N

    N = 1
    if len(input_shape) > 1:
        N = input_shape[-2]

    n_in = input_shape[-1]
    n_out = out_shape[-1]

    return out_shape, {
        'matrix_mult' : N * N * (n_in * n_out * 2),
        'bias'  : N * N * n_out
    }

def diagbiasdense(input_shape, layer):
    out_shape = layer.call(tf.zeros(input_shape)).shape

    N = input_shape[-2]

    n_out = out_shape[-1]

    _, dense_flops = dense(input_shape, layer)
    dense_flops.update({
        'diag_bias' : N * n_out
    })
    return out_shape, dense_flops

def log_layer(input_shape, layer):
    return input_shape, {}

def calc_flops(model, input_shape):
    info_dict = {
        layers.InputLayer : input_layer,
        InnerProduct : pelican_layer,
        Lineq2v2 : pelican_layer,
        Lineq2v0 : pelican_layer,
        DiagBiasDense: diagbiasdense,
        layers.Activation: activation,
        layers.Dense: dense,
        LogLayer: pelican_layer,
        ScalingLayer: pelican_layer
    }

    total = {}
    for n, layer in enumerate(model.layers):
        input_shape, flops = info_dict[type(layer)](input_shape, layer)
        total[f'{n}:{type(layer)}'] = flops

    return total

def get_flops_activ(input_shape, activation):
    # https://github.com/bb511/deepsets_synth/blob/main/fast_deepsets/util/flops.py
    # Taken from above github 20.05.2024
    """Approximates the number of floating point operations in an activation.

    According to https://stackoverflow.com/q/41251698 tanh has 20-100 FLOPs.
    According to https://discourse.julialang.org/t/how-many-flops-does-it-take-to-compute-a-square-root/89027/3,
    exponential uses around 20
    """
    if isinstance(input_shape, Iterable):
        ninputs = functools.reduce(lambda x, y: x * y, input_shape)
    else:
        ninputs = input_shape

    switcher = {
        "relu": lambda: ninputs,
        "tanh": lambda: ninputs * 50,
        "linear": lambda: 0,
        "softmax": lambda: ninputs * (6 * 20 + 6)
    }

    activation_flops = switcher.get(activation, lambda: None)()
    if activation_flops == None:
        raise RuntimeError(f"Number of flops calc. not implemented for {activation}.")

    return activation_flops
