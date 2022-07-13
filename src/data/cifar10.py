import tensorflow as tf
from src.utils.common import get_dataset
from src.data import AsbDataset

DEFAULT_BATCH_SIZE = 32
DEFAULT_IMAGE_SIZE = (32, 32)
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class Cifar10(AsbDataset):
    def __init__(self, image_size=DEFAULT_IMAGE_SIZE, batch_size=DEFAULT_BATCH_SIZE, map_fn=None):
        super(Cifar10, self).__init__(name='cifar10', classes=CLASS_NAMES, image_size=image_size, batch_size=batch_size, map_fn=map_fn)

    def _load_dataset(self, image_size, batch_size, map_fn):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            directory=get_dataset('cifar10/train'),
            labels='inferred',
            label_mode='int',
            batch_size=batch_size,
            image_size=image_size,
            interpolation='nearest'
        )

        test_ds = tf.keras.utils.image_dataset_from_directory(
            directory=get_dataset('cifar10/test'),
            labels='inferred',
            label_mode='int',
            batch_size=batch_size,
            image_size=image_size,
            interpolation='nearest'
        )

        if map_fn is not None:
            train_ds = train_ds.map(map_fn).prefetch(tf.data.AUTOTUNE)
            test_ds = test_ds.map(map_fn).prefetch(tf.data.AUTOTUNE)

        train_ds_size = train_ds.cardinality().numpy()
        train_ds = train_ds.skip(train_ds_size / 10)
        val_ds = train_ds.take(train_ds_size / 10)
        return train_ds, val_ds, test_ds
