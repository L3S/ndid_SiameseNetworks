import tensorflow as tf


class OfflineTripletLoss(tf.keras.losses.Loss):

    def __init__(self, margin=1):
        self.MARGIN = margin
        super().__init__()

    def call(self, distance_AP, distance_AN, margin=None):
        """
        Implements the triplet loss formula from https://towardsdatascience.com/image-similarity-using-triplet-loss-3744c0f67973
        :param distance_AP: distance between anchor and postive sample
        :param distance_AN: distance between anchor and negative sample
        :return:
        """
        if not margin:
            margin = self.MARGIN

        d_AP = tf.math.square(tf.convert_to_tensor(distance_AP))
        d_AN = tf.math.square(tf.convert_to_tensor(distance_AN))

        loss = tf.maximum(0.0, d_AP - d_AN + margin)
        loss = tf.reduce_mean(loss)
        return loss
