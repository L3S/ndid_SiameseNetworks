from common import get_modeldir, get_logdir, target_shape
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


class AlexNet:
    def __init__(self):
        super(AlexNet, self).__init__()

        self.model = None

    def get_model(self):
        if self.model is None:
            self.model = models.Sequential([
                layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=target_shape + (3,)),
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
                layers.Dropout(0.5),

                layers.Dense(10, activation='softmax')
            ])
        return self.model

    def train_model(self, train_ds, validation_ds, test_ds):
        tensorboard_cb = callbacks.TensorBoard(get_logdir("alexnet/fit"))

        # optimizer='adam', SGD W
        self.get_model()
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])
        self.model.summary()
        self.model.fit(train_ds, epochs=50, validation_data=validation_ds, validation_freq=1, callbacks=[tensorboard_cb])
        self.model.evaluate(test_ds)

    def save_model(self, name):
        self.model.save(get_modeldir(name))

    def load_model(self, name):
        self.model = models.load_model(get_modeldir(name))
