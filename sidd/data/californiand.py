import tensorflow as tf
from glob import glob
from sidd.utils.common import get_dataset
from sidd.data import AbsDataset
import numpy as np
from sidd.utils.plot_dataset import plot_sample, plot_label

DEFAULT_BATCH_SIZE = 32
DEFAULT_IMAGE_SIZE = (1024, 768)

def read_labels():
    cm = np.load(str(get_dataset('californiaND')) + '/Correlation matrices/gt_all.npy')
    num_entries = cm.shape[0]
    cm = (cm > 0)
    labels = np.zeros(num_entries)
    class_label = 1
    for row_id in range(num_entries):
        if labels[row_id] != 0:
            assigned_label = labels[row_id]
        else:
            assigned_label = class_label
            class_label += 1
        labels[cm[row_id]] = assigned_label
    return labels

class CaliforniaND(AbsDataset):
    def __init__(self, image_size=DEFAULT_IMAGE_SIZE, batch_size=DEFAULT_BATCH_SIZE, map_fn=None):
        super(CaliforniaND, self).__init__(name='californiand', classes=[], image_size=image_size, batch_size=batch_size, map_fn=map_fn)

    def get_classes(self):
        raise NotImplementedError()

    def _load_dataset(self, image_size, batch_size, map_fn):
        def load(path):
            image_raw = tf.io.read_file(path)
            decoded = tf.image.decode_jpeg(image_raw, channels=3)
            resized = tf.image.resize(decoded, image_size, method='nearest')

            name = tf.strings.split(path, '/')[-1]
            name = tf.strings.split(name, '.', 1)[0]
            name = tf.strings.to_number(name, tf.int32)
            label = tf.py_function(lambda name: labels[name.numpy() - 1], [name], tf.int32)
            return resized, label

        labels = read_labels()
        dataset_path_glob = glob(str(get_dataset('californiaND')) + '/Photos/*.jpg')
        ds = tf.data.Dataset.from_tensor_slices(dataset_path_glob).map(load)

        if batch_size is not None:
            ds = ds.batch(batch_size)

        if map_fn is not None:
            ds = ds.map(map_fn).prefetch(tf.data.AUTOTUNE)
    
        ds_size = ds.cardinality().numpy()

        train_ds = ds.take(ds_size * 0.6)
        val_ds = ds.skip(ds_size * 0.6).take(ds_size * 0.2)
        test_ds = ds.skip(ds_size * 0.6).skip(ds_size * 0.2)
        return train_ds, val_ds, test_ds

if __name__ == "__main__":
    ds = CaliforniaND().get_combined()

    plot_sample(ds)
    plot_label(ds, 93)
