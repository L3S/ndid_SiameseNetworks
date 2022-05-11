import tensorflow as tf

BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)

NUM_CLASSES = 10
CLASS_NAMES = ['fish', 'dog', 'player', 'saw', 'building', 'music', 'truck', 'gas', 'ball', 'parachute']


def load_dataset(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, preprocess_fn=None):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory='../datasets/imagenette2/train/',
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=image_size,
        interpolation='nearest'
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory='../datasets/imagenette2/val/',
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        interpolation='nearest'
    )

    if preprocess_fn is not None:
        train_ds = train_ds.map(preprocess_fn).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.map(preprocess_fn).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds


def load_dataset3(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, preprocess_fn=None):
    train_ds, test_ds = load_dataset(image_size=image_size, batch_size=batch_size, preprocess_fn=preprocess_fn)

    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
    val_ds = test_ds.take(test_ds_size / 2)
    test_ds = test_ds.skip(test_ds_size / 2)

    if True:
        print("Imagenette dataset loaded")
        print("Training data size:", tf.data.experimental.cardinality(train_ds).numpy())
        print("Validation data size:", tf.data.experimental.cardinality(val_ds).numpy())
        print("Evaluation data size:", tf.data.experimental.cardinality(test_ds).numpy())
    return train_ds, val_ds, test_ds
