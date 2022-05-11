from src.utils.common import *
import tensorflow_hub as hub
from tensorflow.keras import layers, callbacks, datasets, Sequential

tensorboard_cb = callbacks.TensorBoard(get_logdir("vit/fit"))

MODEL_URL = "https://tfhub.dev/sayakpaul/vit_s16_fe/1"
MODEL_INPUT_SIZE = [None, 224, 224, 3]

class VitModel(Sequential):
    def __init__(self):
        super(VitModel, self).__init__([
            hub.KerasLayer(MODEL_URL, trainable=False)  # EfficientNet V2 S backbone, frozen weights
        ])
        self.build(MODEL_INPUT_SIZE)

    def compile(self, metrics=['accuracy'], **kwargs):
        super().compile(metrics=metrics, **kwargs)

    def fit(self, x=None, y=None, callbacks=[tensorboard_cb], **kwargs):
        return super().fit(x=x, y=y, callbacks=callbacks, **kwargs)
