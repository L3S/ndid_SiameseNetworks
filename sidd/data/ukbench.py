import math
import tensorflow as tf
from glob import glob
from sidd.utils.common import get_dataset
from sidd.data import AbsDataset

DEFAULT_BATCH_SIZE = 32
DEFAULT_IMAGE_SIZE = (640, 480)


class UKBench(AbsDataset):
    def __init__(self, image_size=DEFAULT_IMAGE_SIZE, batch_size=DEFAULT_BATCH_SIZE, map_fn=None):
        super(UKBench, self).__init__(name='ukbench', classes=[], image_size=image_size, batch_size=batch_size, map_fn=map_fn)

    def get_classes(self):
        raise NotImplementedError()

    def _load_dataset(self, image_size, batch_size, map_fn):
        def load(path):
            image_raw = tf.io.read_file(path)
            decoded = tf.image.decode_jpeg(image_raw, channels=3)
            resized = tf.image.resize(decoded, image_size, method='nearest')

            label = tf.strings.split(path, 'ukbench')[2]  # remove 'ukbench'
            label = tf.strings.split(label, '.', 1)[0]
            label = tf.strings.to_number(label, tf.int32)
            # use the same label for all images of each kind
            label = tf.math.floordiv(label, 4)
            return resized, label

        dataset_path_glob = glob(str(get_dataset('ukbench')) + '/*.jpg')
        ds = tf.data.Dataset.from_tensor_slices(dataset_path_glob).map(load)

        if batch_size is not None:
            ds = ds.batch(batch_size)

        if map_fn is not None:
            ds = ds.map(map_fn).prefetch(tf.data.AUTOTUNE)
    
        ds_size = ds.cardinality().numpy()
        self.num_classes = math.floor(ds_size / 4)

        train_ds = ds.take(ds_size * 0.6)
        val_ds = ds.skip(ds_size * 0.6).take(ds_size * 0.2)
        test_ds = ds.skip(ds_size * 0.6).skip(ds_size * 0.2)
        return train_ds, val_ds, test_ds
