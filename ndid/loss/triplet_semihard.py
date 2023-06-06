import tensorflow as tf

from ndid.loss import _metric_learning


class TripletSemiHardLoss(tf.keras.losses.Loss):
    """Computes the triplet loss with semi-hard negative mining.

    The loss encourages the positive distances (between a pair of embeddings
    with the same labels) to be smaller than the minimum negative distance
    among which are at least greater than the positive distance plus the
    margin constant (called semi-hard negative) in the mini-batch.
    If no such negative exists, uses the largest negative distance instead.
    See: https://arxiv.org/abs/1503.03832.

    We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
    `[batch_size]` of multi-class integer labels. And embeddings `y_pred` must be
    2-D float `Tensor` of l2 normalized embedding vectors.

    Args:
      margin: Float, margin term in the loss definition. Default value is 1.0.
      distance_metric: `str` or a `Callable` that determines distance metric.
          Valid strings are "L2" for l2-norm distance,
          "squared-L2" for squared l2-norm distance,
          and "angular" for cosine similarity.

          A `Callable` should take a batch of embeddings as input and
          return the pairwise distance matrix.
    """

    def __init__(self, margin = 1.0, distance_metric = "L2"):
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def call(self, y_true, y_pred):
        r"""Computes the triplet loss with semi-hard negative mining.

        Args:
        y_true: 1-D integer `Tensor` with shape `[batch_size]` of multiclass integer labels.
        y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should be l2 normalized.

        Returns:
        triplet_loss: float scalar with dtype of `y_pred`.
        """
        labels = tf.convert_to_tensor(y_true, name="labels")
        embeddings = tf.convert_to_tensor(y_pred, name="embeddings")

        convert_to_float32 = (
            embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16
        )
        precise_embeddings = (
            tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings
        )

        # Reshape label tensor to [batch_size, 1].
        lshape = tf.shape(labels)
        labels = tf.reshape(labels, [lshape[0], 1])

        # Build pairwise squared distance matrix

        if self.distance_metric == "L2":
            pdist_matrix = _metric_learning.pairwise_distance(
                precise_embeddings, squared=False
            )

        elif self.distance_metric == "squared-L2":
            pdist_matrix = _metric_learning.pairwise_distance(
                precise_embeddings, squared=True
            )

        elif self.distance_metric == "angular":
            pdist_matrix = _metric_learning.angular_distance(precise_embeddings)

        else:
            pdist_matrix = self.distance_metric(precise_embeddings)

        # Build pairwise binary adjacency matrix.
        adjacency = tf.math.equal(labels, tf.transpose(labels))
        # Invert so we can select negatives only.
        adjacency_not = tf.math.logical_not(adjacency)

        batch_size = tf.size(labels)

        # Compute the mask.
        pdist_matrix_tile = tf.tile(pdist_matrix, [batch_size, 1])
        mask = tf.math.logical_and(
            tf.tile(adjacency_not, [batch_size, 1]),
            tf.math.greater(
                pdist_matrix_tile, tf.reshape(tf.transpose(pdist_matrix), [-1, 1])
            ),
        )
        mask_final = tf.reshape(
            tf.math.greater(
                tf.math.reduce_sum(
                    tf.cast(mask, dtype=tf.dtypes.float32), 1, keepdims=True
                ),
                0.0,
            ),
            [batch_size, batch_size],
        )
        mask_final = tf.transpose(mask_final)

        adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
        mask = tf.cast(mask, dtype=tf.dtypes.float32)

        # negatives_outside: smallest D_an where D_an > D_ap.
        negatives_outside = tf.reshape(
            _metric_learning.masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size]
        )
        negatives_outside = tf.transpose(negatives_outside)

        # negatives_inside: largest D_an.
        negatives_inside = tf.tile(
            _metric_learning.masked_maximum(pdist_matrix, adjacency_not), [1, batch_size]
        )
        semi_hard_negatives = tf.where(mask_final, negatives_outside, negatives_inside)

        loss_mat = tf.math.add(self.margin, pdist_matrix - semi_hard_negatives)

        mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
            tf.ones([batch_size])
        )

        # In lifted-struct, the authors multiply 0.5 for upper triangular
        #   in semihard, they take all positive pairs except the diagonal.
        num_positives = tf.math.reduce_sum(mask_positives)

        triplet_loss = tf.math.truediv(
            tf.math.reduce_sum(
                tf.math.maximum(tf.math.multiply(loss_mat, mask_positives), 0.0)
            ),
            num_positives,
        )

        if convert_to_float32:
            return tf.cast(triplet_loss, embeddings.dtype)
        else:
            return triplet_loss
