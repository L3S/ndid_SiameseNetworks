import tensorflow as tf
from tensorflow.keras import layers, callbacks, Model, Sequential
from src.utils.common import get_logdir

tensorboard_cb = callbacks.TensorBoard(get_logdir('vgg16/fit'))

BATCH_SIZE = 32
TARGET_SHAPE = (224, 224)

PRETRAIN_EPOCHS = 20
EMBEDDING_VECTOR_DIMENSION = 4096

class VGG16Model(Model):
    def __init__(self, input_shape=TARGET_SHAPE, weights=None, **kwargs):
        if weights == "imagenet":
            core = tf.keras.applications.VGG16(
                include_top=False,
                input_shape=(224, 224, 3),
                weights="imagenet",
                # pooling="avg",
            )
            core.summary()
            core.trainable = False

            model = Sequential([
                core,
                layers.Flatten(),
                layers.Dense(4096, activation='relu'),
                layers.Dense(4096, activation='relu'),
                layers.Dense(10, activation='softmax')
            ])
        else:
            model = tf.keras.applications.VGG16(
                include_top=True,
                input_shape=input_shape + (3,),
                weights=None,
                classes=10,
                **kwargs
            )

        super(VGG16Model, self).__init__(inputs=model.input, outputs=model.output, name='vgg16')

    def compile(self,
                optimizer=tf.keras.optimizers.RMSprop(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'],
                **kwargs):
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    def fit(self, x=None, y=None, batch_size=None, epochs=PRETRAIN_EPOCHS, callbacks=[tensorboard_cb], **kwargs):
        return super().fit(x=x, y=y, batch_size=batch_size, epochs=epochs, callbacks=callbacks, **kwargs)

    def get_embedding_model(self):
        core = Model(inputs=self.input, outputs=self.layers[-2].output, name=self.name + '_emb')
        for layer in core.layers: layer.trainable = False
        return core

    @staticmethod
    def preprocess_input(image, label):
        image = tf.keras.applications.vgg16.preprocess_input(image)
        return image, label
