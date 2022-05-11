import tensorflow as tf
from tensorflow.keras import layers, backend


# Provided two tensors t1 and t2
# Euclidean distance = sqrt(sum(square(t1-t2)))
def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

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


"""
### Contrastive Loss

$ Loss = Y*Dist(v_1,v_2)^2 + (1-Y)*max(margin-D,0)^2$

$Y$ is the GT target (1 if $v_1$ and $v_2$ belong to the same class, 0 otherwise). If images are from the same class, use the squared distance as loss (you want to push the distance to be close to 0 for same-class couples), otherwise keep the (squared) maximum between 0 and $margin - D$.

For different-class couples, the distance should be pushed to a high value. The **margin identifies a cone inside which vectors are considered the same**. For cosine distance, which has range [0,2], **1 is NOT an adequate value**).

**NOTE** In the loss implementation below, we calculate the mean of the two terms, though this should not actually be necessary (the minimizer value for the loss is the same whether the loss is divided by 2 or not).
"""
def ContrastiveLoss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        'constrastive_loss' function with data ('margin') attached.
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

        Arguments:
            y_true: List of labels (1 for same-class pair, 0 for different-class), fp32.
            y_pred: List of predicted distances, fp32.

        Returns:
            A tensor containing constrastive loss as floating point value.
        """

        square_dist = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean((1 - y_true) * square_dist + (y_true) * margin_square)

    return contrastive_loss
