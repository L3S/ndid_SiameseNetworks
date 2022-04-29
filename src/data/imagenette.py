import tensorflow as tf


def normalize(image, label):
    # image = tf.cast(image, tf.uint8)
    # image = tf.image.per_image_standardization(image)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    # image9 = (image / (255 / 2)) - 1
    return image, label


def load_dataset():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory='../datasets/imagenette2/train/',
        labels='inferred',
        label_mode='int',
        batch_size=32,
        image_size=(224, 224),
        interpolation='nearest'
    ).map(normalize).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory='../datasets/imagenette2/val/',
        labels='inferred',
        label_mode='int',
        batch_size=32,
        image_size=(224, 224),
        shuffle=False,
        interpolation='nearest'
    ).map(normalize).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds
