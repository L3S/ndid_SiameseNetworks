import tensorflow as tf


class TripletEasyLoss(tf.keras.losses.Loss):
    """
    Implements the triplet loss formula from
    https://towardsdatascience.com/image-similarity-using-triplet-loss-3744c0f67973

    :param distance_pos: distance between anchor and positive sample
    :param distance_neg: distance between anchor and negative sample
    :return:
    """

    def __init__(self, margin=1):
        super().__init__()
        self.margin = margin

    def call(self, y_true, y_pred):
        r"""Computes the triplet loss with random pair mining.

        Args:
        y_true: 1-D integer `Tensor` with shape `[batch_size]` of multiclass integer labels.
        y_pred: 2-D float `Tensor` of embedding vectors. Embeddings should be l2 normalized.

        Returns:
        triplet_loss: float scalar with dtype of `y_pred`.
        """

        y_true = tf.math.square(tf.convert_to_tensor(y_true))
        y_pred = tf.math.square(tf.convert_to_tensor(y_pred))

        loss = tf.maximum(0.0, y_true - y_pred + self.margin)
        loss = tf.reduce_mean(loss)
        return loss
