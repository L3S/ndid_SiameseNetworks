import tensorflow as tf

BATCH_SIZE = 32
IMAGE_SIZE = (32, 32)

NUM_CLASSES = 10
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_dataset(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, map_fn=None):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory='../datasets/cifar10/train/',
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=image_size,
        interpolation='nearest'
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory='../datasets/cifar10/test/',
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        interpolation='nearest'
    )

    if map_fn is not None:
        train_ds = train_ds.map(map_fn).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.map(map_fn).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds


def load_dataset3(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, map_fn=None):
    train_ds, test_ds = load_dataset(image_size=image_size, batch_size=batch_size, map_fn=map_fn)

    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    train_ds = train_ds.skip(train_ds_size / 10)
    val_ds = train_ds.take(train_ds_size / 10)

    if True:
        print("CIFAR10 dataset loaded")
        print("Training data size:", tf.data.experimental.cardinality(train_ds).numpy())
        print("Validation data size:", tf.data.experimental.cardinality(val_ds).numpy())
        print("Evaluation data size:", tf.data.experimental.cardinality(test_ds).numpy())
    return train_ds, val_ds, test_ds
