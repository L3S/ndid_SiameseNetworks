import tensorflow as tf


class OfflineTripletLoss(tf.keras.losses.Loss):

    def __init__(self, margin=1):
        super().__init__()
        self.margin = margin

    def call(self, distance_pos, distance_neg):
        """
        Implements the triplet loss formula from
        https://towardsdatascience.com/image-similarity-using-triplet-loss-3744c0f67973

        :param distance_pos: distance between anchor and positive sample
        :param distance_neg: distance between anchor and negative sample
        :return:
        """

        distance_pos = tf.math.square(tf.convert_to_tensor(distance_pos))
        distance_neg = tf.math.square(tf.convert_to_tensor(distance_neg))

        loss = tf.maximum(0.0, distance_pos - distance_neg + self.margin)
        loss = tf.reduce_mean(loss)
        return loss
