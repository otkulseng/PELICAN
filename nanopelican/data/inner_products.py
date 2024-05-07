import tensorflow as tf
from keras import backend as K
import functools

def get_flops_activ(input_shape, activation):
    # from https://github.com/bb511/deepsets_synth/tree/main/fast_deepsets
    """Approximates the number of floating point operations in an activation.

    According to https://stackoverflow.com/q/41251698 tanh has 20-100 FLOPs.
    """
    if isinstance(input_shape, list):
        ninputs = functools.reduce(lambda x, y: x * y, input_shape)
    else:
        ninputs = input_shape

    switcher = {
        ""
        "relu": lambda: ninputs,
        "tanh": lambda: ninputs * 50,
        "linear": lambda: 0,
    }

    activation_flops = switcher.get(activation, lambda: None)()
    if activation_flops == None:
        raise RuntimeError(f"Number of flops calc. not implemented for {activation}.")

    return activation_flops

def get_flops(data_format, input_shape):

    if data_format.lower() in ['epxpypz', 'pxpypze']:
        # Input_shape is num_particles x 4
        num_particles, _ = input_shape
        # 4 multiplications, 4 additions per element in
        # square matrix
        return num_particles**2 * (4 + 4)

    raise TypeError(
        f"Could not find flops for data_format: {data_format}"
    )



def get_instantons(data_format, dtype=tf.float32):
    my_dict = {
        'epxpypz': tf.constant([[1, 0, 0, -1],
                                [1, 0, 0, 1]], dtype=dtype),
        'pxpypze': tf.constant([[-1, 0, 0, 1],
                                [1, 0, 0, 1]], dtype=dtype)
    }

    key = data_format.lower()

    if type(key) == str and key in my_dict:
        return my_dict[key]

    raise TypeError(
        f"Could not find instantons for data_format: {data_format}"
    )

def get_handler(data_format):
    if callable(data_format):
        return data_format

    my_dict = {
        'epxpypz': inner_prods_from_Epxpypz,
        'pxpypze': inner_prods_from_inverted_Epxpypz
    }


    key = data_format.lower()
    if type(key) == str and key in my_dict:
        return my_dict[key]

    raise TypeError(
        f"Could not interpret data handler: {data_format}"
    )


def inner_prods_from_Epxpypz(data):
    """Assumes data is of shape Batch x num_particles x 4
    where the last axis are (E, px, py, pz)
    """
    M = tf.linalg.diag(tf.constant([1, -1, -1, -1], dtype=tf.float32))
    return tf.einsum("...pi, ij, ...qj->...pq", data, M, data)


def inner_prods_from_inverted_Epxpypz(data):
    """Assumes data is of shape Batch x num_particles x 4
    where the last axis are (px, py, pz, E)
    """
    M = tf.linalg.diag(tf.constant([-1, -1, -1, 1], dtype=tf.float32))
    return tf.einsum("...pi, ij, ...qj->...pq", data, M, data)

