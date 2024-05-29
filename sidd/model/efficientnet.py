import tensorflow as tf
from tensorflow.keras import layers, callbacks, Model, Sequential
from sidd.utils.common import get_logdir

tensorboard_cb = callbacks.TensorBoard(get_logdir('efficientnet/fit'))

BATCH_SIZE = 32
TARGET_SHAPE = (384, 384)

PRETRAIN_EPOCHS = 20
EMBEDDING_VECTOR_DIMENSION = 1280


class EfficientNetModel(Model):
    def __init__(self, input_shape=TARGET_SHAPE, num_classes=10, weights=None, **kwargs):
        if weights == "imagenet":
            model = tf.keras.applications.EfficientNetV2S(
                include_top=True,
                input_shape=input_shape + (3,),
                weights="imagenet",
            )
            model.trainable = False
        elif weights == "finetune":
            core = tf.keras.applications.EfficientNetV2S(
                include_top=False,
                input_shape=input_shape + (3,),
                weights="imagenet",
            )
            core.trainable = False

            model = Sequential([
                core,
                layers.GlobalAveragePooling2D(name="avg_pool"),
                layers.Dropout(0.2, name="top_dropout"),
                layers.Dense(num_classes, activation='softmax', kernel_initializer={
                    "class_name": "VarianceScaling",
                    "config": {
                        "scale": 1.0 / 3.0,
                        "mode": "fan_out",
                        "distribution": "uniform",
                    },
                }, bias_initializer=tf.constant_initializer(0), name="predictions"),
            ])
        else:
            model = tf.keras.applications.EfficientNetV2S(
                include_top=True,
                input_shape=input_shape + (3,),
                weights=None,
                classes=num_classes,
                **kwargs
            )

        super(EfficientNetModel, self).__init__(inputs=model.input, outputs=model.output, name='efficientnet')

    def compile(self,
                optimizer=None,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'],
                train_size=None,
                **kwargs):

        if optimizer is None and train_size is not None:
            pretrain_steps = PRETRAIN_EPOCHS * train_size
            optimizer = tf.keras.optimizers.RMSprop(tf.keras.optimizers.schedules.CosineDecay(1e-3, pretrain_steps))
        elif optimizer is None:
            optimizer = tf.keras.optimizers.RMSprop()

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
        image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
        return image, label
