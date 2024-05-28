import tensorflow as tf

def repeat_const(tensor, myconst):
    # https://stackoverflow.com/questions/68345125/how-to-concatenate-a-tensor-to-a-keras-layer-along-batch-without-specifying-bat
    shapes = tf.shape(tensor)
    return tf.repeat(myconst, shapes[0], axis=0)