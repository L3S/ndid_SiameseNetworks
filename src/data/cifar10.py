import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt
from src.utils.common import get_datadir, process_images, process_images_couple
import tensorflow as tf


def load_dataset():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)
    val = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)
    return train, val


def cifar10_complete():
    train, val = load_dataset()
    return train.concatenate(val)


def cifar10_complete_resized():
    ds = cifar10_complete()
    return ds.map(process_images_couple).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


