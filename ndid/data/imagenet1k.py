import os

import tensorflow as tf
import tensorflow_datasets as tfds
from ndid.data import AsbDataset
from ndid.utils.common import get_dataset


class ImageNet1k(AsbDataset):
    def __init__(self, map_fn=None, batch_size=None):
        super(ImageNet1k, self).__init__(name='imagenet1k', num_classes=1000, batch_size=batch_size, map_fn=map_fn)

    def _load_dataset(self, image_size, batch_size, map_fn):
        builder = tfds.builder('imagenet2012')

        download_config = tfds.download.DownloadConfig(
            manual_dir=str(get_dataset('imagenet'))
        )

        builder.download_and_prepare(download_config=download_config)

        train_ds = builder.as_dataset(split=tfds.Split.TRAIN, batch_size=batch_size, as_supervised=True)
        test_ds = builder.as_dataset(split=tfds.Split.TEST, batch_size=batch_size, as_supervised=True)
        val_ds = builder.as_dataset(split=tfds.Split.VALIDATION, batch_size=batch_size, as_supervised=True)
        assert isinstance(train_ds, tf.data.Dataset)
        assert isinstance(test_ds, tf.data.Dataset)
        assert isinstance(val_ds, tf.data.Dataset)
        return train_ds, val_ds, test_ds
