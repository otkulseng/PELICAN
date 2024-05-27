
from keras import layers
from nanopelican.layers import *
import tensorflow as tf

def input_layer(input_shape, layer):
    return input_shape, {}

def inner_product(input_shape, layer):
    out_shape = layer.call(tf.zeros(input_shape)).shape
    return out_shape, {}

def lineq2v2(input_shape, layer):
    out_shape = layer.call(tf.zeros(input_shape)).shape
    return out_shape, {

    }

def lineq2v0(input_shape, layer):
    out_shape = layer.call(tf.zeros(input_shape)).shape
    return out_shape, {}


def activation(input_shape, layer):
    out_shape = layer.call(tf.ones(input_shape)).shape
    return out_shape, {}

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



def calc_flops(model, input_shape):
    info_dict = {
        layers.InputLayer : input_layer,
        InnerProduct : inner_product,
        Lineq2v2 : lineq2v2,
        Lineq2v0 : lineq2v0,
        DiagBiasDense: diagbiasdense,
        layers.Activation: activation,
        layers.Dense: dense
    }

    total = {}
    for layer in model.layers:
        input_shape, flops = info_dict[type(layer)](input_shape, layer)
        total[str(type(layer))] = flops

    return total
