import tensorflow as tf
from keras import backend as K


def interpret_string(data_format):

    data_format = data_format.rstrip("]").split("[")
    if len(data_format)> 1:
        data_format[1] = data_format[1].split(":")
        data_format[1] = [int(elem) if elem != "" else None for elem in data_format[1]]

    if len(data_format) == 1:
        return data_format[0], (None, None)
    return data_format[0], tuple(data_format[1])


def get_handler(data_format):
    if callable(data_format):
        return data_format

    function, _ = interpret_string(data_format)

    my_dict = {
        'fourvec': inner_prods_from_Epxpypz,
        'inverted': inner_prods_from_inverted_Epxpypz
    }

    if type(function) == str and function in my_dict:
        return my_dict[data_format]

    raise TypeError(
        f"Could not interpret data handler: {function}"
    )

def get_indeces(data_format):
    _, indeces = interpret_string(data_format)
    return indeces


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

