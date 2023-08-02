import tensorflow as tf

from sidd.loss import _metric_learning


class TripletHardLoss(tf.keras.losses.Loss):
    """Computes the triplet loss with hard negative and hard positive mining.

    The loss encourages the maximum positive distance (between a pair of embeddings
    with the same labels) to be smaller than the minimum negative distance plus the
    margin constant in the mini-batch.
    The loss selects the hardest positive and the hardest negative samples
    within the batch when forming the triplets for computing the loss.
    See: https://arxiv.org/pdf/1703.07737.

    We expect labels `y_true` to be provided as 1-D integer `Tensor` with shape
    `[batch_size]` of multi-class integer labels. And embeddings `y_pred` must be
    2-D float `Tensor` of l2 normalized embedding vectors.

    Args:
      margin: Float, margin term in the loss definition. Default value is 1.0.
      soft: Boolean, if set, use the soft margin version. Default value is False.
      distance_metric: `str` or a `Callable` that determines distance metric.
          Valid strings are "L2" for l2-norm distance,
          "squared-L2" for squared l2-norm distance,
          and "angular" for cosine similarity.

          A `Callable` should take a batch of embeddings as input and
          return the pairwise distance matrix.
    """

    def __init__(self, margin = 1.0, soft = False, distance_metric = "L2"):
        super().__init__()
        self.margin = margin
        self.soft = soft
        self.distance_metric = distance_metric

    def call(self, y_true, y_pred):
        r"""Computes the triplet loss with hard negative and hard positive mining.

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

        # Build pairwise squared distance matrix.
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

        adjacency_not = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
        # hard negatives: smallest D_an.
        hard_negatives = _metric_learning.masked_minimum(pdist_matrix, adjacency_not)

        batch_size = tf.size(labels)

        adjacency = tf.cast(adjacency, dtype=tf.dtypes.float32)

        mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
            tf.ones([batch_size])
        )

        # hard positives: largest D_ap.
        hard_positives = _metric_learning.masked_maximum(pdist_matrix, mask_positives)

        if self.soft:
            triplet_loss = tf.math.log1p(tf.math.exp(hard_positives - hard_negatives))
        else:
            triplet_loss = tf.maximum(hard_positives - hard_negatives + self.margin, 0.0)

        # Get final mean triplet loss
        triplet_loss = tf.reduce_mean(triplet_loss)

        if convert_to_float32:
            return tf.cast(triplet_loss, embeddings.dtype)
        else:
            return triplet_loss
