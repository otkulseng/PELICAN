#!/usr/bin/env python
from nanopelican.layers import *
from nanopelican.scripts import run
from keras import layers, Model
from qkeras import *

def qmlp(units, activs, nbits, quantizer, x, batchnorm=False, mask=None, n_int=0):
    for unit, act in zip(units, activs):
        if batchnorm:
            x = QBatchNormalization(
                beta_quantizer=quantizer,
                gamma_quantizer=quantizer,
                mean_quantizer=quantizer,
                variance_quantizer=quantizer,
            )(x)
        x = QDense(
            units=unit,
            kernel_quantizer=quantizer,
            bias_quantizer=quantizer,
        )(x)

        if mask is not None:
            x = layers.Multiply()([x, mask])

        if act != 'linear':
            x = QActivation(
                activation=format_qactivation(act, nbits, n_int=n_int)
            )(x)
    return x


def CreateModel(shape, conf):

    x = x_in = layers.Input(shape, name='input')

    x, _ = InnerProduct(conf['inner_product'], name='inner_product')(x)

    x = Lineq2v2(units=conf['n_hidden'], symmetric=True, hollow=True)(x)

    x = layers.Activation(activation=conf['activation'])(x)

    x = Lineq2v0(units=conf['n_out'])(x)

    return Model(inputs=x_in, outputs=x)



# These functions are taken from:
# https://github.com/bb511/deepsets_synth/blob/main/fast_deepsets/deepsets/deepsets_quantised.py
# 03.06.2024
def format_quantiser(n_bits, n_int=0):


    """Format the quantisation of the ml floats in a QKeras way."""
    if n_bits == 1:
        return "binary(alpha=1)"
    elif n_bits == 2:
        return "ternary(alpha=1)"
    else:
        # A number is then represented as
        # (+-)n_int.(n_bits - n_int - 1)
        # i.e.
        # 1 bit for sign
        # n_int for integer
        # the rest, n_bits - n_int - 1 for decimal
        return f"quantized_bits({n_bits}, {n_int}, alpha=1)"


def format_qactivation(activation, n_bits, n_int=0):
    """Format the activation function strings in a QKeras friendly way."""
    return f"quantized_{activation}({n_bits}, {n_int})"

def main():
    run(CreateModel)
if __name__ == '__main__':
    main()