import tensorflow as tf
from tensorflow.keras import layers, callbacks, Sequential, Model

from ndid.model import AsbModel
from ndid.utils.common import get_logdir, get_weightsdir

tensorboard_cb = callbacks.TensorBoard(get_logdir('alexnet/fit'))

BATCH_SIZE = 32
TARGET_SHAPE = (224, 224)

PRETRAIN_EPOCHS = 50
EMBEDDING_VECTOR_DIMENSION = 4096

def create_alexnet_model(input_shape, num_classes):
    return Sequential([
        layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

        layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

        layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'),
        layers.BatchNormalization(),

        layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'),
        layers.BatchNormalization(),

        layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'),
        layers.BatchNormalization(),

        layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),

        layers.Flatten(),

        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(4096, activation='relu'),
        layers.Dropout(rate=0.5),

        layers.Dense(name='predictions', units=num_classes, activation='softmax')
    ])


class AlexNetModel(Model):

    def __init__(self, input_shape=TARGET_SHAPE, classes=10, weights="imagenet", train_size=None, **kwargs):
        if weights == "imagenet":
            core = create_alexnet_model(
                input_shape=input_shape + (3,),
                num_classes=1000,
            )
            core.load_weights(get_weightsdir('alexnet_imagenet'))
            core.trainable = False
            core.summary()

            core = Model(inputs=core.input, outputs=core.layers[-2].output)
            
            core.summary()
            model = Sequential([
                core,
                layers.Dropout(rate=0.5),
                layers.Dense(name='predictions', units=classes, activation='softmax'),
            ])
            model.summary()
        else:
            model = create_alexnet_model(
                input_shape=input_shape + (3,),
                num_classes=classes,
            )

        super(AlexNetModel, self).__init__(inputs=model.input, outputs=model.output, name='alexnet')

    def compile(self,
                optimizer=tf.optimizers.SGD(learning_rate=0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'],
                **kwargs):
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    def fit(self, x=None, y=None, batch_size=None, epochs=PRETRAIN_EPOCHS, callbacks=[tensorboard_cb], **kwargs):
        return super().fit(x=x, y=y, batch_size=batch_size, epochs=epochs, callbacks=callbacks, **kwargs)

    def get_embedding_model(self):
        core = Model(inputs=self.input, outputs=self.layers[-2].output, name=self.name + '_emb')
        for layer in core.layers:
            layer.trainable = False
        return core

    @staticmethod
    def get_target_shape():
        return TARGET_SHAPE

    @staticmethod
    def preprocess_input(image, label):
        # Normalize images to have a mean of 0 and standard deviation of 1
        image = tf.image.per_image_standardization(image)
        # Resize images to 277x277
        image = tf.image.resize(image, TARGET_SHAPE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return image, label
