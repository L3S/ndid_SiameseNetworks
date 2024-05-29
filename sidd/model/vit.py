from sidd.utils.common import *
import tensorflow_hub as hub
from tensorflow.keras import layers, callbacks, datasets, Model, Sequential

tensorboard_cb = callbacks.TensorBoard(get_logdir("vit/fit"))

BATCH_SIZE = 256
TARGET_SHAPE = (224, 224)

MODEL_URL = "https://tfhub.dev/sayakpaul/vit_s16_fe/1"


class VitModel(Model):
    def __init__(self, input_shape=TARGET_SHAPE, num_classes=10, weights=None, **kwargs):
        if weights == "imagenet":
            model = Sequential([hub.KerasLayer(MODEL_URL, trainable=False)])
            model.build((None,) + TARGET_SHAPE + (3,))
            super(VitModel, self).__init__(inputs=model.input, outputs=model.output, name='vit')
        else:
            raise ValueError("Unknown weights: %s" % weights)

    def fit(self, x=None, y=None, callbacks=[tensorboard_cb], **kwargs):
        return super().fit(x=x, y=y, callbacks=callbacks, **kwargs)

    def get_embedding_model(self):
        return self

    @staticmethod
    def get_target_shape():
        return TARGET_SHAPE

    @staticmethod
    def preprocess_input(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, TARGET_SHAPE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = (image - 0.5) * 2  # ViT requires images in range [-1,1]
        return image, label
