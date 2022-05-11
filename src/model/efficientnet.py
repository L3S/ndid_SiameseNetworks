from src.utils.common import *
import tensorflow_hub as hub
from tensorflow.keras import layers, callbacks, datasets, Sequential

tensorboard_cb = callbacks.TensorBoard(get_logdir("efficientnet/fit"))

MODEL_URL = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/feature_vector/2"
MODEL_INPUT_SIZE = [None, 384, 384, 3]

class EfficientNetModel(Sequential):
    def __init__(self):
        super(EfficientNetModel, self).__init__([
            hub.KerasLayer(MODEL_URL, trainable=False)  # EfficientNet V2 S backbone, frozen weights
        ])
        self.build(MODEL_INPUT_SIZE)

    def compile(self, metrics=['accuracy'], **kwargs):
        super().compile(metrics=metrics, **kwargs)

    def fit(self, x=None, y=None, callbacks=[tensorboard_cb], **kwargs):
        return super().fit(x=x, y=y, callbacks=callbacks, **kwargs)
