# https://github.com/bb511/deepsets_synth/blob/main/fast_deepsets/util/flops.py
# Taken from above github 20.05.2024

# Calculates the number of FLOPs in the DeepSets models.
import numpy as np
import functools

import tensorflow as tf
from tensorflow import keras


def get_flops_dense(input_shape, units):
    """Calculate the number of floating point operations in a dense layer."""
    if isinstance(input_shape, list):
        MAC = functools.reduce(lambda x, y: x * y, input_shape) * units
    else:
        MAC = input_shape*units

    # Add biases.
    ADD = units

    return MAC * 2 + ADD


def get_flops_activ(input_shape, activation):
    """Approximates the number of floating point operations in an activation.

    According to https://stackoverflow.com/q/41251698 tanh has 20-100 FLOPs.
    """
    if isinstance(input_shape, list):
        ninputs = functools.reduce(lambda x, y: x * y, input_shape)
    else:
        ninputs = input_shape

    switcher = {
        "relu": lambda: ninputs,
        "tanh": lambda: ninputs * 50,
        "linear": lambda: 0,
        None: lambda: 0
    }

    activation_flops = switcher.get(activation, lambda: None)()
    if activation_flops == None:
        raise RuntimeError(f"Number of flops calc. not implemented for {activation}.")

    return activation_flops
