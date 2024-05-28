import tensorflow as tf
from glob import glob
from sidd.utils.common import get_dataset
from sidd.data import AbsDataset
import numpy as np
import os.path

from sidd.utils.plot_dataset import plot_label, plot_sample

DEFAULT_BATCH_SIZE = 32
DEFAULT_IMAGE_SIZE = (500, 500)

def read_labels():
    path = str(get_dataset('mirflickr')) + '/labels.txt'
    if os.path.isfile(path):
        return np.loadtxt(path, dtype=int)
    else:
        class_label = 1
        labels = np.zeros((1000000), dtype=int)

        for nd_file in ['duplicates', 'IND_clusters', 'NIND_clusters']:
            file = open(str(get_dataset('mirflickr-full')) + '/' + nd_file + ".txt")

            for line in file:
                line = line.strip('\n ').split(' ')
                duplicates = np.array(line, dtype=int)
                if (labels[duplicates] != np.zeros_like(duplicates)).all():
                    assigned_label = np.amin(labels[duplicates])
                    labels_to_update = set(labels[duplicates])
                    labels_to_update.remove(assigned_label)
                    for label in labels_to_update:
                        labels[labels==label] = assigned_label
                else:
                    assigned_label = class_label
                    class_label += 1
                labels[duplicates] = assigned_label

        max_labels = np.amax(labels) 

        unassigned_entries = (labels[labels==0]).shape[0]
        add_labels = np.arange(max_labels+1, max_labels+unassigned_entries+1)
        labels[labels==0] = add_labels

        labels = labels[0:25000]
        np.savetxt(path, labels, fmt='%d')
        return labels


class MFND(AbsDataset):
    def __init__(self, image_size=DEFAULT_IMAGE_SIZE, batch_size=DEFAULT_BATCH_SIZE, map_fn=None):
        super(MFND, self).__init__(name='mirflickr', classes=[], image_size=image_size, batch_size=batch_size, map_fn=map_fn)

    def get_classes(self):
        raise NotImplementedError()

    def get_combined(self):
        return super().get_train()

    def _load_dataset(self, image_size, batch_size, map_fn):
        def load(path):
            image_raw = tf.io.read_file(path)
            decoded = tf.image.decode_jpeg(image_raw, channels=3)
            resized = tf.image.resize(decoded, image_size, method='nearest')

            name = tf.strings.split(path, '/')[-1]
            name = tf.strings.split(name, '.', 1)[0]
            name = tf.strings.substr(name, 2, -1) # remove 'im'
            name = tf.strings.to_number(name, tf.int32)
            label = tf.py_function(lambda name: labels[name.numpy() - 1], [name], tf.int32)
            return resized, label

        labels = read_labels()
        dataset_path_glob = glob(str(get_dataset('mirflickr')) + '/images/*.jpg')
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
    ds = MFND().get_combined()

    #plot_sample(ds)
    plot_label(ds, 4430)
