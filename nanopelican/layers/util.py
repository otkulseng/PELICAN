import tensorflow as tf

def repeat_const(tensor, myconst):
    # https://stackoverflow.com/questions/68345125/how-to-concatenate-a-tensor-to-a-keras-layer-along-batch-without-specifying-bat
    shapes = tf.shape(tensor)
    return tf.repeat(myconst, shapes[0], axis=0)


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

