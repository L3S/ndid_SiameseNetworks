import tensorflow as tf
from ndid.utils.common import get_dataset
from ndid.data import AsbDataset

DEFAULT_BATCH_SIZE = 32
DEFAULT_IMAGE_SIZE = (400, 320)
CLASS_NAMES = ['fish', 'dog', 'player', 'saw', 'building', 'music', 'truck', 'gas', 'ball', 'parachute']


class Imagenette(AsbDataset):
    def __init__(self, image_size=DEFAULT_IMAGE_SIZE, batch_size=DEFAULT_BATCH_SIZE, map_fn=None):
        super(Imagenette, self).__init__(name='imagenette', classes=CLASS_NAMES, image_size=image_size, batch_size=batch_size, map_fn=map_fn)

    def _load_dataset(self, image_size, batch_size, map_fn):
        train_ods = tf.keras.utils.image_dataset_from_directory(
            directory=get_dataset('imagenette2/train'),
            labels='inferred',
            label_mode='int',
            batch_size=batch_size,
            image_size=image_size,
            interpolation='nearest'
        )

        test_ods = tf.keras.utils.image_dataset_from_directory(
            directory=get_dataset('imagenette2/val'),
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
