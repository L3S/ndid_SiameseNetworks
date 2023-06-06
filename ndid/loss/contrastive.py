import tensorflow as tf


class ContrastiveLoss(tf.keras.losses.Loss):
    r"""Computes the contrastive loss between `y_true` and `y_pred`.

    This loss encourages the embedding to be close to each other for
    the samples of the same label and the embedding to be far apart at least
    by the margin constant for the samples of different labels.

    See: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

    We expect labels `y_true` to be provided as 1-D integer `Tensor`
    with shape `[batch_size]` of binary integer labels. And `y_pred` must be
    1-D float `Tensor` with shape `[batch_size]` of distances between two
    embedding matrices.

    The euclidean distances `y_pred` between two embedding matrices
    `a` and `b` with shape `[batch_size, hidden_size]` can be computed
    as follows:

    >>> a = tf.constant([[1, 2],
    ...                 [3, 4],[5, 6]], dtype=tf.float16)
    >>> b = tf.constant([[5, 9],
    ...                 [3, 6],[1, 8]], dtype=tf.float16)
    >>> y_pred = tf.linalg.norm(a - b, axis=1)
    >>> y_pred
    <tf.Tensor: shape=(3,), dtype=float16, numpy=array([8.06 , 2.   , 4.473],
    dtype=float16)>

    <... Note: constants a & b have been used purely for
    example purposes and have no significant value ...>

    Args:
      margin: `Float`, margin term in the loss definition.
        Default value is 1.0.
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply.
        Default value is `SUM_OVER_BATCH_SIZE`.
      name: (Optional) name for the loss.
    """

    def __init__(self, margin = 1.0):
        super().__init__()
        self.margin = margin

    def call(self, y_true, y_pred):    
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.dtypes.cast(y_true, y_pred.dtype)
        return y_true * tf.math.square(y_pred) + (1.0 - y_true) * tf.math.square(
            tf.math.maximum(self.margin - y_pred, 0.0)
        )
