from src.common import *
import tensorflow as tf
from tensorflow.keras import layers, callbacks, datasets, Sequential


class AlexNetModel(Sequential):
    def __init__(self):
        super(AlexNetModel, self).__init__([
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
            layers.Dropout(rate=0.5),

            layers.Dense(name='unfreeze', units=10, activation='softmax')
        ])

    @staticmethod
    def x_dataset():
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

        validation_images, validation_labels = train_images[:5000], train_labels[:5000]
        train_images, train_labels = train_images[5000:], train_labels[5000:]

        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

        train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
        test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
        validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
        print("Training data size:", train_ds_size)
        print("Test data size:", test_ds_size)
        print("Validation data size:", validation_ds_size)

        # plot_grid25(train_ds)
        # plot_grid25(test_ds)
        # plot_grid25(validation_ds)

        train_ds = (train_ds.map(process_images_couple).shuffle(buffer_size=train_ds_size).batch(batch_size=32, drop_remainder=True))
        test_ds = (test_ds.map(process_images_couple).shuffle(buffer_size=train_ds_size).batch(batch_size=32, drop_remainder=True))
        validation_ds = (validation_ds.map(process_images_couple).shuffle(buffer_size=train_ds_size).batch(batch_size=32, drop_remainder=True))
        return train_ds, test_ds, validation_ds

    def x_train(self, train_ds, validation_ds):
        tensorboard_cb = callbacks.TensorBoard(get_logdir("alexnet/fit"))

        # optimizer='adam', SGD W
        self.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])
        self.summary()
        self.fit(train_ds, epochs=50, validation_data=validation_ds, validation_freq=1, callbacks=[tensorboard_cb])
