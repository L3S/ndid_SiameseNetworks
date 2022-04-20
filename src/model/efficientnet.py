from src.utils.common import *
import tensorflow as tf
from tensorflow.keras import layers, callbacks, datasets, Sequential

tensorboard_cb = callbacks.TensorBoard(get_logdir("efficientnet/fit"))


class EfficientNetModel(Sequential):
    def __init__(self):
        super(EfficientNetModel, self).__init__([
            layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                          input_shape=target_shape + (3,)),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

            layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
            layers.BatchNormalization(),
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

            layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            layers.BatchNormalization(),

            layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            layers.BatchNormalization(),

            layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            layers.BatchNormalization(),

            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

            layers.Flatten(),

            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),

            layers.Dense(4096, activation='relu'),
            layers.Dropout(rate=0.5),

            layers.Dense(name='unfreeze', units=10, activation='softmax')
        ])
