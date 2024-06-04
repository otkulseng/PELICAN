#!/usr/bin/env python
from nanopelican.layers import *
from nanopelican.scripts import run
from keras import layers, Model
from qkeras import *

def mlp(units, activs, nbits, quantizer, x, batchnorm=False, mask=None):
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
                activation=format_qactivation(act, nbits)
            )(x)
    return x





def CreateModel(shape, conf):

    QUANT = format_quantiser(
        n_bits=conf['n_bits']
    )

    x = x_in = layers.Input(shape, name='input')
    x, mask = InnerProduct(conf['inner_product'], name='inner_product')(x)

    x = QBatchNormalization(
        beta_quantizer=QUANT,
        gamma_quantizer=QUANT,
        mean_quantizer=QUANT,
        variance_quantizer=QUANT,
        name='2v2bnorm'
    )(x)
    x = Lineq2v2(symmetric=True, hollow=True, num_avg=conf['num_avg'], diag_bias=True, name='2v2')(x)

    x = mlp(
        units=conf['hidden']['units'],
        activs=conf['hidden']['activs'],
        nbits = conf['n_bits'],
        quantizer=QUANT,
        x=x,
        batchnorm=conf['hidden']['batchnorm'],
        mask=mask
    )

    x = QBatchNormalization(
        beta_quantizer=QUANT,
        gamma_quantizer=QUANT,
        mean_quantizer=QUANT,
        variance_quantizer=QUANT,
        name='2v0bnorm'
    )(x)
    x = Lineq2v0(num_avg=conf['num_avg'], name='2v0')(x)

    x = mlp(
        units=conf['out']['units'],
        activs=conf['out']['activs'],
        nbits = conf['n_bits'],
        quantizer=QUANT,
        x=x,
        batchnorm=conf['out']['batchnorm']
    )

    x = layers.Activation(activation=conf['out_activation'])(x)
    return Model(inputs=x_in, outputs=x)



# These functions are taken from:
# https://github.com/bb511/deepsets_synth/blob/main/fast_deepsets/deepsets/deepsets_quantised.py
# 03.06.2024
def format_quantiser(n_bits):
    """Format the quantisation of the ml floats in a QKeras way."""
    if n_bits == 1:
        return "binary(alpha=1)"
    elif n_bits == 2:
        return "ternary(alpha=1)"
    else:
        return f"quantized_bits({n_bits}, 0, alpha=1)"


def format_qactivation(activation, n_bits):
    """Format the activation function strings in a QKeras friendly way."""
    return f"quantized_{activation}({n_bits}, 0)"


def main():
    run(CreateModel)

if __name__ == '__main__':
    main()