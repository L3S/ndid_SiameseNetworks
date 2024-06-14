from abc import ABC, abstractmethod
import tensorflow as tf


class AsbModel(ABC, tf.keras.Model):

    @staticmethod
    @abstractmethod
    def get_target_shape():
        pass

    @staticmethod
    @abstractmethod
    def preprocess_input(image, label):
        pass
