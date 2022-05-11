import tensorflow as tf

BATCH_SIZE = 32
IMAGE_SIZE = (400, 320)

NUM_CLASSES = 3
CLASS_NAMES = ['building', 'dog', 'player']


def load_dataset(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, preprocess_fn=None):
    ds = tf.keras.utils.image_dataset_from_directory(
        directory='../datasets/simple3/',
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=image_size,
        interpolation='nearest'
    )

    if preprocess_fn is not None:
        ds = ds.map(preprocess_fn).prefetch(tf.data.AUTOTUNE)

    return ds


def load_dataset3(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, preprocess_fn=None):
    ds = load_dataset(image_size=image_size, batch_size=batch_size, preprocess_fn=preprocess_fn)

    ds_size = tf.data.experimental.cardinality(ds).numpy()
    train_ds = ds.take(ds_size * 0.7)
    val_ds = ds.skip(ds_size * 0.7).take(ds_size * 0.15)
    test_ds = ds.skip(ds_size * 0.7).skip(ds_size * 0.15)

    if True:
        print("Simple 3 dataset loaded")
        print("Training data size:", tf.data.experimental.cardinality(train_ds).numpy())
        print("Validation data size:", tf.data.experimental.cardinality(val_ds).numpy())
        print("Evaluation data size:", tf.data.experimental.cardinality(test_ds).numpy())
    return train_ds, val_ds, test_ds
