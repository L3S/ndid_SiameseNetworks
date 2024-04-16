import tensorflow as tf
from sidd.utils.common import get_dataset
from sidd.data import AbsDataset

DEFAULT_BATCH_SIZE = 32
DEFAULT_IMAGE_SIZE = (32, 32)
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class Cifar10(AbsDataset):
    def __init__(self, image_size=DEFAULT_IMAGE_SIZE, batch_size=DEFAULT_BATCH_SIZE, map_fn=None):
        super(Cifar10, self).__init__(name='cifar10', classes=CLASS_NAMES, image_size=image_size, batch_size=batch_size, map_fn=map_fn)

    def _load_dataset(self, image_size, batch_size, map_fn):
        train_ods = tf.keras.utils.image_dataset_from_directory(
            directory=get_dataset('cifar10/train'),
            labels='inferred',
            label_mode='int',
            batch_size=batch_size,
            image_size=image_size,
            interpolation='nearest'
        )

        test_ods = tf.keras.utils.image_dataset_from_directory(
            directory=get_dataset('cifar10/test'),
            labels='inferred',
            label_mode='int',
            batch_size=batch_size,
            image_size=image_size,
            interpolation='nearest'
        )

        if map_fn is not None:
            train_ods = train_ods.map(map_fn).prefetch(tf.data.AUTOTUNE)
            test_ods = test_ods.map(map_fn).prefetch(tf.data.AUTOTUNE)

        test_ods_size = test_ods.cardinality().numpy()
        val_ds = test_ods.skip(test_ods_size / 2)
        test_ds = test_ods.take(test_ods_size / 2)
        return train_ods, val_ds, test_ds
