from sidd.utils.common import *
import tensorflow_hub as hub
from tensorflow.keras import layers, callbacks, datasets, Model, Sequential

tensorboard_cb = callbacks.TensorBoard(get_logdir("simclr/fit"))

BATCH_SIZE = 256
TARGET_SHAPE = (224, 224)


class SimclrModel(Model):
    def __init__(self, input_shape=TARGET_SHAPE, num_classes=1000, weights="imagenet", train_size=None, **kwargs):
        if weights == "imagenet":
            self.saved_model = tf.saved_model.load('./models/simclr')
            super(SimclrModel, self).__init__(name='simclr')
        else:
            raise ValueError("Unknown weights: %s" % weights)
        
    def call(self, inputs, training=False):
        return self.saved_model(inputs, training)['logits_sup']

    def get_embedding_model(self):
        return self

    @staticmethod
    def get_target_shape():
        return TARGET_SHAPE

    @staticmethod
    def preprocess_input(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, TARGET_SHAPE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = tf.clip_by_value(image, 0., 1.)
        return image, label
