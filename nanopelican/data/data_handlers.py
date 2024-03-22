import tensorflow as tf
from keras import backend as K

def get_handler(data_format):
    if callable(data_format):
        return data_format

    my_dict = {
        'fourvec': inner_prods_from_Epxpypz
    }

    if type(data_format) == str and data_format in my_dict:
        return my_dict[data_format]

    raise TypeError(
        f"Could not interpret data handler: {data_format}"
    )


def inner_prods_from_Epxpypz(data):
    """Assumes data is of shape Batch x num_particles x 4
    where the last axis are (E, px, py, pz)
    """
    M = tf.linalg.diag(tf.constant([1, -1, -1, -1], dtype=tf.float32))
    return tf.einsum("...pi, ij, ...qj->...pq", data, M, data)


