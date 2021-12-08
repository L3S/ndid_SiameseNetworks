from src.utils.common import *
import tensorflow as tf
from tensorflow.keras import layers, callbacks, datasets, Sequential

tensorboard_cb = callbacks.TensorBoard(get_logdir("alexnet/fit"))

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

    def compile(self, optimizer=tf.optimizers.SGD(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'],
                loss_weights=None, weighted_metrics=None, run_eagerly=None, steps_per_execution=None, **kwargs):
        super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, **kwargs)

    def fit(self, x=None, y=None, batch_size=None, epochs=50, verbose='auto', callbacks=[tensorboard_cb], validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1,
            max_queue_size=10, workers=1, use_multiprocessing=False):
        return super().fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle,
                           class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps,
                           validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)

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
