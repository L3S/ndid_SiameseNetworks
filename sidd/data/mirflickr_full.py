import tensorflow as tf
from glob import glob
from sidd.utils.common import get_dataset
from sidd.data import AbsDataset
from sidd.utils.plot_dataset import plot_sample, plot_label
import numpy as np
import os.path
from tqdm import tqdm

DEFAULT_BATCH_SIZE = 32
DEFAULT_IMAGE_SIZE = (500, 500)

# Regarding MirFlickr Labels, there are three files:
#   - duplicates - contains identical images
#   - IND-clusters (Identical Near Duplicates): Near duplicates which are derived from the same image through transformation (cropping, rotation, rescaling etc.)
#   - NIND-clusters (Non-Identical Near Duplicates): share same content but differ in illumination, subject movement, viewpoint.
#
# You can find below a code snippet which makes 'class'-Labels out of these files.

def read_labels():
    path = str(get_dataset('mirflickr-full')) + '/labels.txt'
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

        np.savetxt(path, labels, fmt='%d')
        return labels

class MFNDFull(AbsDataset):
    def __init__(self, image_size=DEFAULT_IMAGE_SIZE, batch_size=DEFAULT_BATCH_SIZE, map_fn=None):
        super(MFNDFull, self).__init__(name='mirflickr_full', classes=[], image_size=image_size, batch_size=batch_size, map_fn=map_fn)

    def get_classes(self):
        raise NotImplementedError()

    def _load_dataset(self, image_size, batch_size, map_fn):
        def load(path):
            image_raw = tf.io.read_file(path)
            decoded = tf.image.decode_jpeg(image_raw, channels=3, try_recover_truncated=True, acceptable_fraction=0.5)
            resized = tf.image.resize(decoded, image_size, method='nearest')

            name = tf.strings.split(path, '/')[-1]
            name = tf.strings.split(name, '.', 1)[0]
            name = tf.strings.to_number(name, tf.int32)
            label = tf.py_function(lambda name: labels[name.numpy()], [name], tf.int32)
            return resized, label

        labels = read_labels()
        dataset_path_glob = glob(str(get_dataset('mirflickr-full')) + '/images/*/*.jpg')
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
    ds = MFNDFull().get_combined()

    #plot_sample(ds)
    plot_label(ds, 9267)
    #plot_label(ds, 126479)

    #for path, label in tqdm(ds.unbatch()):
        #print(path)
    # path = '/home/astappiev/nsir/datasets/mirflickr-full/images/68/686806.jpg'
    # image_raw = tf.io.read_file(path)
    # decoded = tf.image.decode_jpeg(image_raw, channels=3, try_recover_truncated=True, acceptable_fraction=0.5)
    # resized = tf.image.resize(decoded, DEFAULT_IMAGE_SIZE, method='nearest')
