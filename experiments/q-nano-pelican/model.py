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

def mlp(units, activs,x,batchnorm=False, mask=None):
    for unit, act in zip(units, activs):
        if batchnorm:
            x = BatchNormalization()(x)

        x = layers.Dense(unit, act)(x)

        if mask is not None:
            x = layers.Multiply()([x, mask])

    return x



def CreateNano(shape, conf):
    x = x_in = layers.Input(shape, name='input')
    x, mask = InnerProduct(conf['inner_product'], name='inner_product')(x)

    x = layers.BatchNormalization()(x)

    x = Lineq2v2(symmetric=True, hollow=True, diag_bias=True,num_avg=conf['2v2'], name='2v2')(x)

    x = mlp(
        units=conf['hidden']['units'],
        activs=conf['hidden']['activs'],
        x=x,
    )

    x = layers.BatchNormalization()(x)
    x = Lineq2v0(num_avg=conf['2v0'], name='2v0')(x)

    x = mlp(
        units=conf['out']['units'],
        activs=conf['out']['activs'],
        x=x,
    )

    x = layers.Activation(activation=conf['out_activation'])(x)
    return Model(inputs=x_in, outputs=x)
def CreateModel(shape, conf):
    if not conf['quantize']:
        return CreateNano(shape, conf)


    if 'inp' in conf:
        inp_quantizer = get_quantizer(format_quantiser(
            n_bits=conf['inp']['n_bits'],
            n_int=conf['inp']['n_int']
        ))

    if 'internal' in conf:
        internal = get_quantizer(format_quantiser(
            n_bits=conf['internal']['n_bits'],
            n_int=conf['internal']['n_int']
        ))

    QUANT = format_quantiser(
        n_bits=conf['n_bits'],
        n_int=conf['n_int']
    )

    x = x_in = layers.Input(shape, name='input')

    if 'inp' in conf:
        x = inp_quantizer(x)
    x, mask = InnerProduct(conf['inner_product'], name='inner_product')(x)
    # if 'inp' in conf:
    #     x = inp_quantizer(x)

    x = QBatchNormalization(
        beta_quantizer=QUANT,
        gamma_quantizer=QUANT,
        mean_quantizer=QUANT,
        variance_quantizer=QUANT,
        name='2v2bnorm'
    )(x)

    if 'internal' in conf:
        x = internal(x)

    x = Lineq2v2(symmetric=True, hollow=True, diag_bias=True,num_avg=conf['2v2'], name='2v2')(x)

    if 'internal' in conf:
        x = internal(x)

    x = qmlp(
        units=conf['hidden']['units'],
        activs=conf['hidden']['activs'],
        nbits = conf['n_bits'],
        quantizer=QUANT,
        x=x,
        # mask=mask
        n_int=conf['n_int']
    )

    if 'internal' in conf:
        x = internal(x)
    x = QBatchNormalization(
        beta_quantizer=QUANT,
        gamma_quantizer=QUANT,
        mean_quantizer=QUANT,
        variance_quantizer=QUANT,
        name='2v0bnorm'
    )(x)
    if 'internal' in conf:
        x = internal(x)
    x = Lineq2v0(num_avg=conf['2v0'], name='2v0')(x)

    if 'internal' in conf:
        x = internal(x)

    x = qmlp(
        units=conf['out']['units'],
        activs=conf['out']['activs'],
        nbits = conf['n_bits'],
        quantizer=QUANT,
        x=x,
        n_int=conf['n_int']
    )

    x = layers.Activation(activation=conf['out_activation'])(x)
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