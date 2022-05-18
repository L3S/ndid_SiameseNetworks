import tensorflow as tf
from tensorflow.keras import layers, backend


def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    # Provided two tensors t1 and t2
    # Euclidean distance = sqrt(sum(square(t1-t2)))
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, backend.epsilon()))


def cosine_distance(vects):
    """Find the Cosine distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """
    # NOTE: Cosine_distance = 1 - cosine_similarity
    # Cosine distance is defined betwen [0,2] where 0 is vectors with the same direction and verse,
    # 1 is perpendicular vectors and 2 is opposite vectors
    cosine_similarity = layers.Dot(axes=1, normalize=True)(vects)
    return 1 - cosine_similarity
