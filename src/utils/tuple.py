import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt
from src.utils.common import get_datadir, resize_image, normalize_image
import tensorflow as tf

target_shape = (227, 227)


def process_images(image):
    # Resize images from 32x32 to 277x277
    image = resize_image(image, target_shape)
    image = normalize_image(image)
    return image


def shuffle_arrays(arrays, set_seed=-1):
    """Shuffles arrays in-place, in the same order, along axis=0

    Parameters:
    -----------
    arrays : List of NumPy arrays.
    set_seed : Seed value if int >= 0, else seed is random.
    """
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    seed = np.random.randint(0, 2 ** (32 - 1) - 1) if set_seed < 0 else set_seed

    for arr in arrays:
        rstate = np.random.RandomState(seed)
        rstate.shuffle(arr)


def produce_tuples(images, labels):
    unique_labels = np.unique(labels, return_counts=True)
    total_labels = len(unique_labels[0])
    images_per_label = np.max(unique_labels[1])
    tuples_per_label = int(images_per_label / 3)
    total_tuples = int(tuples_per_label * total_labels)

    image_shape = images.shape[1]

    # re arrange whole dataset by labels
    labels_stat = np.zeros((total_labels), dtype='uint16')
    labels_train = np.empty((total_labels, images_per_label, image_shape), dtype='uint8')

    for i in range(len(images)):
        tr_leb = labels[i]
        tr_img = images[i]
        labels_train[tr_leb, labels_stat[tr_leb]] = tr_img
        labels_stat[tr_leb] += 1

    # shuffle all
    for i in range(total_labels):
        np.random.shuffle(labels_train[i])

    # create tuples
    anchor_images = np.empty((total_tuples, image_shape), dtype='uint8')
    anchor_labels = np.empty((total_tuples), dtype='uint8')

    for i in range(total_labels):
        for j in range(tuples_per_label):
            anchor_labels[i * tuples_per_label + j] = i
            anchor_images[i * tuples_per_label + j] = labels_train[i, j]

    positive_images = np.empty((total_tuples, image_shape), dtype='uint8')
    positive_labels = np.empty((total_tuples), dtype='uint8')

    for i in range(total_labels):
        for j in range(tuples_per_label):
            positive_labels[i * tuples_per_label + j] = i
            positive_images[i * tuples_per_label + j] = labels_train[i, tuples_per_label + j]

    negative_images = np.empty((total_tuples, image_shape), dtype='uint8')
    negative_labels = np.empty((total_tuples), dtype='uint8')

    for i in range(total_labels):
        for j in range(tuples_per_label):
            negative_labels[i * tuples_per_label + j] = i
            negative_images[i * tuples_per_label + j] = labels_train[i, tuples_per_label * 2 + j]

    # we need to ensure we use random labels, but without images from anchor label
    shuffle_arrays([negative_labels, negative_images])

    for i in range(total_labels):
        k = ((i + 1) * tuples_per_label, 0)[i == total_labels - 1]
        for j in range(tuples_per_label):
            c = i * tuples_per_label + j
            tmp_label = negative_labels[c]

            if tmp_label == i:
                tmp_image = negative_images[c]
                while negative_labels[k] == i:
                    k += 1
                negative_labels[c] = negative_labels[k]
                negative_images[c] = negative_images[k]
                negative_labels[k] = tmp_label
                negative_images[k] = tmp_image

    # randomize them one more time
    for i in range(total_labels):
        shuffle_arrays([
            negative_labels[i * tuples_per_label:(i + 1) * tuples_per_label],
            negative_images[i * tuples_per_label:(i + 1) * tuples_per_label]
        ])

    return (anchor_images, anchor_labels), (positive_images, positive_labels), (negative_images, negative_labels)


def save_tuples(anchor_images, anchor_labels, positive_images, positive_labels, negative_images, negative_labels):
    data = [anchor_images, anchor_labels, positive_images, positive_labels, negative_images, negative_labels]

    with open(get_datadir('cifar10_tuples.pkl'), 'wb') as outfile:
        pickle.dump(data, outfile, -1)


def load_tuples():
    with open(get_datadir('cifar10_tuples.pkl'), 'rb') as infile:
        result = pickle.load(infile)

    return (result[0], result[1]), (result[2], result[3]), (result[4], result[5])


def prepare_dataset():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    (anchor_images, anchor_labels), (positive_images, positive_labels), (negative_images, negative_labels) = produce_tuples(train_images + test_images, train_labels + test_labels)

    anchor_ds = tf.data.Dataset.from_tensor_slices(anchor_images)
    positive_ds = tf.data.Dataset.from_tensor_slices(positive_images)
    negative_ds = tf.data.Dataset.from_tensor_slices(negative_images)

    anchor_ds = (anchor_ds.map(process_images).batch(batch_size=32, drop_remainder=True))
    positive_ds = (positive_ds.map(process_images).batch(batch_size=32, drop_remainder=True))
    negative_ds = (negative_ds.map(process_images).batch(batch_size=32, drop_remainder=True))

    dataset = tf.data.Dataset.zip((anchor_ds, positive_ds, negative_ds))
    # dataset = dataset.shuffle(buffer_size=1024)
    return dataset


def visualize(anchor, positive, negative):
    """Visualize a few triplets from the supplied batches."""

    def show(ax, image):
        ax.imshow(image)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    fig = plt.figure(figsize=(9, 9))

    axs = fig.subplots(3, 3)
    for i in range(3):
        show(axs[i, 0], anchor[0][i])
        show(axs[i, 1], positive[0][i])
        show(axs[i, 2], negative[0][i])
    plt.show()
