import tensorflow as tf
from tensorflow.keras import layers, callbacks, Model, Sequential
from src.utils.common import get_logdir

tensorboard_cb = callbacks.TensorBoard(get_logdir('mobilenet/fit'))

BATCH_SIZE = 32
TARGET_SHAPE = (224, 224)

PRETRAIN_EPOCHS = 20
EMBEDDING_VECTOR_DIMENSION = 1024

class MobileNetModel(Model):
    def __init__(self):
        core = tf.keras.applications.MobileNet(
            include_top=False,
        )

        x = core.output
        x = layers.GlobalAveragePooling2D(keepdims=True)(x)
        x = layers.Dropout(1e-3, name='dropout')(x)
        x = layers.Conv2D(10, (1, 1), padding='same', name='conv_preds')(x)
        x = layers.Reshape((10,), name='reshape_2')(x)
        x = layers.Activation(activation='softmax', name='predictions')(x)

        x = tf.keras.applications.mobilenet.preprocess_input(x)
        super(MobileNetModel, self).__init__(inputs=core.input, outputs=x)

    def compile(self,
                optimizer=tf.keras.optimizers.RMSprop(),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'],
                **kwargs):
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    def fit(self, x=None, y=None, batch_size=None, epochs=PRETRAIN_EPOCHS, callbacks=[tensorboard_cb], **kwargs):
        return super().fit(x=x, y=y, batch_size=batch_size, epochs=epochs, callbacks=callbacks, **kwargs)

    def get_embedding_model(self):
        core = Model(inputs=self.input, outputs=self.layers[-7].output)
        return Sequential([
            core,
            layers.Flatten(),
        ])

    @staticmethod
    def preprocess_input(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.mobilenet.preprocess_input(image)
        return image, label
