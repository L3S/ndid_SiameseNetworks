from tensorflow.keras import Model
from abc import ABC, abstractmethod


class AsbModel(ABC, Model):
    def get_classes(self):
        return self.classes

    @staticmethod
    @abstractmethod
    def get_target_shape():
        pass

    @staticmethod
    @abstractmethod
    def preprocess_input(image, label):
        pass
