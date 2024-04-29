import tensorflow as tf
from keras import backend as K

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

