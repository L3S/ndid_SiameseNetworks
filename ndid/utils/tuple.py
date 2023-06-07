import numpy as np


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
    labels_train = np.empty((total_labels, images_per_label, image_shape), dtype='uint32')

    for i in range(len(images)):
        tr_leb = labels[i]
        tr_img = images[i]
        labels_train[tr_leb, labels_stat[tr_leb]] = tr_img
        labels_stat[tr_leb] += 1

    # shuffle all
    for i in range(total_labels):
        np.random.shuffle(labels_train[i])

    # create tuples
    anchor_images = np.empty((total_tuples, image_shape), dtype='uint32')
    anchor_labels = np.empty((total_tuples), dtype='uint32')

    for i in range(total_labels):
        for j in range(tuples_per_label):
            anchor_labels[i * tuples_per_label + j] = i
            anchor_images[i * tuples_per_label + j] = labels_train[i, j]

    positive_images = np.empty((total_tuples, image_shape), dtype='uint32')
    positive_labels = np.empty((total_tuples), dtype='uint32')

    for i in range(total_labels):
        for j in range(tuples_per_label):
            positive_labels[i * tuples_per_label + j] = i
            positive_images[i * tuples_per_label + j] = labels_train[i, tuples_per_label + j]

    negative_images = np.empty((total_tuples, image_shape), dtype='uint32')
    negative_labels = np.empty((total_tuples), dtype='uint32')

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
