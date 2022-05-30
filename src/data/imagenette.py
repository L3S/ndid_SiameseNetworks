import tensorflow as tf
from src.data import AsbDataset

DEFAULT_BATCH_SIZE = 32
DEFAULT_IMAGE_SIZE = (400, 320)
CLASS_NAMES = ['fish', 'dog', 'player', 'saw', 'building', 'music', 'truck', 'gas', 'ball', 'parachute']


class Imagenette(AsbDataset):
    def __init__(self, image_size=DEFAULT_IMAGE_SIZE, batch_size=DEFAULT_BATCH_SIZE, map_fn=None):
        super(Imagenette, self).__init__(name='imagenette', classes=CLASS_NAMES, image_size=image_size, batch_size=batch_size, map_fn=map_fn)

    def _load_dataset(self, image_size, batch_size, map_fn):
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

        if map_fn is not None:
            train_ds = train_ds.map(map_fn).prefetch(tf.data.AUTOTUNE)
            test_ds = test_ds.map(map_fn).prefetch(tf.data.AUTOTUNE)

        test_ds_size = test_ds.cardinality().numpy()
        val_ds = test_ds.take(test_ds_size / 2)
        test_ds = test_ds.skip(test_ds_size / 2)
        return train_ds, val_ds, test_ds
