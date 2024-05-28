import tensorflow as tf
from keras import layers
from .utils import *

class InnerProduct(layers.Layer):
    def __init__(self, arg_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.arg_dict = arg_dict
        self.data_handler = get_handler(arg_dict['data_format'])
        self.use_spurions = arg_dict['spurions']

        if self.use_spurions:
            self.spurions = get_spurions(arg_dict['data_format'])
            self.spurions = tf.expand_dims(self.spurions, axis=0)

    def calc_flops(self, input_shape):
        # assumes num_particles x num_features
        N = input_shape[-2]
        F = input_shape[-1]
        if self.use_spurions:
            N += len(self.spurions)

        input_shape = (N, F)

        return get_flops(self.arg_dict['data_format'])(input_shape)


    def build(self, input_shape):
        return super().build(input_shape)

    def compute_output_shape(self, input_shape):
        # if inp is (B, N, custom)
        N = input_shape[-2]

        if self.use_spurions:
            return (input_shape[0], N+self.spurions.shape[-2], N+self.spurions.shape[-2], 1)

        return (input_shape[0], N, N, 1)

    def call(self, inputs):
        # Assumes input_shape is
        # Batch x num_particles (padded) x CUSTOM
        # where self.data_handler is supposed to convert
        # to the inner products Batch x num_particles x num_particles
        # # Add spurions

        if self.use_spurions:
            spurions = layers.Lambda(lambda x: repeat_const(x, self.spurions))(inputs)
            inputs = layers.Concatenate(axis=-2)([inputs, spurions])

        # TODO: Quantize Bits Here!!

        inner_prods = self.data_handler(inputs)
        # inner_prods = tf.math.log(1 + inner_prods)


        return tf.expand_dims(inner_prods, axis=-1)




def ptetaphi_spurions(dtype=None):
    """
    For massless particles,
        E   = p_T cosh(eta)
        p_z = p_T sinh(eta)

    """

    # Set eta big, not necessary to set to inf
    eta = 1.0e1
    phi = 0.0
    p_T = 2.0*tf.exp(-eta)

    return tf.stack([
        [p_T, eta, phi],
        [p_T, -eta, phi]
    ], axis=0)

def get_spurions(data_format, dtype=tf.dtypes.float32):
    my_dict = {
        'epxpypz': tf.constant([[1, 0, 0, -1],
                                [1, 0, 0, 1]], dtype=dtype),
        'pxpypze': tf.constant([[-1, 0, 0, 1],
                                [1, 0, 0, 1]], dtype=dtype),
        'ptetaphi': ptetaphi_spurions(dtype=dtype)
    }

    key = data_format.lower()

    if type(key) == str and key in my_dict:
        return my_dict[key]

    raise TypeError(
        f"Could not find spurions for data_format: {data_format}"
    )


def fourvec_flops(input_shape):

    # Assumes shape N x 4
    N = input_shape[-2]
    F = input_shape[-1]
    assert(F == 4)

    # For each inner product (N**2 in total) between, say, p and q
    # do: 0 + p_0 * q_0 - p_1 * p_1 - p_2 * p_2 - p_3 * p_3

    flops = {
        'inner-products': N**2 * (4 + 4)
    }
    return flops

def ptetaphi_flops(input_shape):
    # assumes shape N x 3
    N = input_shape[-2]
    F = input_shape[-1]
    assert(F == 3)

    # pt_matr = tf.einsum('...p, ...q->...pq', pt, pt)
    # eta_matr = tf.expand_dims(eta, axis=-1) - tf.expand_dims(eta, axis=-2)
    # phi_matr = tf.expand_dims(phi, axis=-1) - tf.expand_dims(phi, axis=-2)

    # # * is elementwise (hadamard)
    # return pt_matr * (tf.cosh(eta_matr) - tf.cos(phi_matr))

    flops = {
        'pt-outer-product': N**2,
        'eta-sum': 2 * N**2,
        'phi-sum': 2 * N**2,
        'cosh': N**2 * 40, # see flops.py. exp use around 20, so would be 20 + 20 = 40
        'cos':  N**2 * 40, # see https://latkin.org/blog/2014/11/09/a-simple-benchmark-of-various-math-operations/
        'diff': 2 * N**2,
        'prod': N**2
    }
    return flops



def get_flops(data_format):
    my_dict = {
        'epxpypz': fourvec_flops,
        'pxpypze': fourvec_flops,
        'ptetaphi': ptetaphi_flops
    }

    key = data_format.lower()
    if type(key) == str and key in my_dict:
        return my_dict[key]

    raise TypeError(
        f"Could not interpret data handler: {data_format}"
    )


def get_handler(data_format):
    if callable(data_format):
        return data_format

    my_dict = {
        'epxpypz': inner_prods_from_Epxpypz,
        'pxpypze': inner_prods_from_inverted_Epxpypz,
        'ptetaphi': inner_prods_from_ptetaphi
    }


    key = data_format.lower()
    if type(key) == str and key in my_dict:
        return my_dict[key]

    raise TypeError(
        f"Could not interpret data handler: {data_format}"
    )


def inner_prods_from_Epxpypz(data):
    """Assumes data is of shape Batch x num_particles x 4
    where the last axis is (E, px, py, pz)
    """

    # OPTIM: can optimizize this by symmetry and hollowness
    M = tf.linalg.diag(tf.constant([1, -1, -1, -1], dtype=tf.dtypes.float32))
    return tf.einsum("...pi, ij, ...qj->...pq", data, M, data)


def inner_prods_from_inverted_Epxpypz(data):
    """Assumes data is of shape Batch x num_particles x 4
    where the last axis is (px, py, pz, E)
    """
    M = tf.linalg.diag(tf.constant([-1, -1, -1, 1], dtype=tf.dtypes.float32))
    return tf.einsum("...pi, ij, ...qj->...pq", data, M, data)

def inner_prods_from_ptetaphi(data):
    """Assumes data is of shape ... x num_particles x 3
    where last axis is (pt, eta, phi)
    """

    # All of these are now ... x num_particles
    pt = data[..., 0]
    eta = data[..., 1]
    phi = data[..., 2]

    pt_matr = tf.einsum('...p, ...q->...pq', pt, pt)
    eta_matr = tf.expand_dims(eta, axis=-1) - tf.expand_dims(eta, axis=-2)
    phi_matr = tf.expand_dims(phi, axis=-1) - tf.expand_dims(phi, axis=-2)

    # * is elementwise (hadamard)
    return pt_matr * (tf.cosh(eta_matr) - tf.cos(phi_matr))




