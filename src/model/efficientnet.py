from src.utils.common import *
import tensorflow_hub as hub
from tensorflow.keras import layers, callbacks, datasets, Sequential

tensorboard_cb = callbacks.TensorBoard(get_logdir("efficientnet/fit"))

BATCH_SIZE = 256
TARGET_SHAPE = (384, 384)

MODEL_URL = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/feature_vector/2"

class EfficientNetModel(Sequential):
    def __init__(self):
        super(EfficientNetModel, self).__init__([
            hub.KerasLayer(MODEL_URL, trainable=False)  # EfficientNet V2 S backbone, frozen weights
        ])
        self.build((None,) + TARGET_SHAPE + (3,))

    def compile(self, metrics=['accuracy'], **kwargs):
        super().compile(metrics=metrics, **kwargs)

    def fit(self, x=None, y=None, callbacks=[tensorboard_cb], **kwargs):
        return super().fit(x=x, y=y, callbacks=callbacks, **kwargs)

    @staticmethod
    def preprocess_input(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, TARGET_SHAPE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return image, label
