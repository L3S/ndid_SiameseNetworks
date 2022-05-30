import tensorflow as tf
from src.data import AsbDataset

DEFAULT_BATCH_SIZE = 6
DEFAULT_IMAGE_SIZE = (400, 320)
CLASS_NAMES = ['building', 'dog', 'player']


class Simple3(AsbDataset):
    def __init__(self, image_size=DEFAULT_IMAGE_SIZE, batch_size=DEFAULT_BATCH_SIZE, map_fn=None):
        super(Simple3, self).__init__(name='simple3', classes=CLASS_NAMES, image_size=image_size, batch_size=batch_size, map_fn=map_fn)

    def _load_dataset(self, image_size, batch_size, map_fn):
        ds = tf.keras.utils.image_dataset_from_directory(
            directory='../datasets/simple3/',
            labels='inferred',
            label_mode='int',
            batch_size=batch_size,
            image_size=image_size,
            interpolation='nearest'
        )

        if map_fn is not None:
            ds = ds.map(map_fn).prefetch(tf.data.AUTOTUNE)

        ds_size = ds.cardinality().numpy()
        train_ds = ds.take(ds_size * 0.6)
        val_ds = ds.skip(ds_size * 0.6).take(ds_size * 0.2)
        test_ds = ds.skip(ds_size * 0.6).skip(ds_size * 0.2)
        return train_ds, val_ds, test_ds
