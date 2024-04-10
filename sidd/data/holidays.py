import tensorflow as tf
from glob import glob
from sidd.utils.common import get_dataset
from sidd.data import AsbDataset

DEFAULT_BATCH_SIZE = 32
DEFAULT_IMAGE_SIZE = (286, 294)


class Holidays(AsbDataset):
    def __init__(self, image_size=DEFAULT_IMAGE_SIZE, batch_size=DEFAULT_BATCH_SIZE, map_fn=None):
        super(Holidays, self).__init__(name='holidays', classes=[], image_size=image_size, batch_size=batch_size, map_fn=map_fn)

    def get_classes(self):
        raise NotImplementedError()

    def get_num_classes(self):
        raise NotImplementedError()

    def get_train(self):
        raise NotImplementedError()

    def get_val(self):
        raise NotImplementedError()

    def get_test(self):
        raise NotImplementedError()

    def get_combined(self):
        return super().get_train()

    def _load_dataset(self, image_size, batch_size, map_fn):
        def load(path):
            image_raw = tf.io.read_file(path)
            decoded = tf.image.decode_jpeg(image_raw, channels=3)
            resized = tf.image.resize(decoded, image_size, method='nearest')

            label = tf.strings.split(path, '/')[-1]
            label = tf.strings.split(label, '.', 1)[0]
            label = tf.strings.to_number(label, tf.int32)
            return resized, label

        dataset_path_glob = glob(str(get_dataset('holidays')) + '/*.jpg')
        train_ds = tf.data.Dataset.from_tensor_slices(dataset_path_glob).map(load)

        if batch_size is not None:
            train_ds = train_ds.batch(batch_size)

        if map_fn is not None:
            train_ds = train_ds.map(map_fn).prefetch(tf.data.AUTOTUNE)

        return train_ds, None, None
