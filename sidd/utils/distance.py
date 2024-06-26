import tensorflow as tf
from tensorflow.keras import layers, backend


def euclidean_distance(vector):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vector: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    # Provided two tensors t1 and t2
    # Euclidean distance = sqrt(sum(square(t1-t2)))
    x, y = vector
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, backend.epsilon()))


def cosine_distance(vector):
    """Find the Cosine distance between two vectors.

    Arguments:
        vector: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """
    # NOTE: Cosine_distance = 1 - cosine_similarity
    # Cosine distance is defined between [0,2] where 0 is vectors with the same direction and verse,
    # 1 is perpendicular vectors and 2 is opposite vectors
    return 1 - layers.Dot(axes=1, normalize=True)(vector)
