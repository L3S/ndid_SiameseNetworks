import tensorflow as tf
from os import path, curdir
import time
import matplotlib.pyplot as plt

# buffer_size = 5000
buffer_size = 50000
target_shape = (227, 227)
CIFAR10_CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def process_images_couple(image, label):
    return process_images(image), label


def process_images(image):
    # Normalize images to have a mean of 0 and standard deviation of 1
    # image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, target_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.uint8)
    return image


def subplot_image(nrows, ncols, index, image, title = None):
    ax = plt.subplot(nrows, ncols, index)
    ax.imshow(image)
    if title is not None:
        ax.set_title(title)
    ax.axis('off')


def plot_grid25(dataset):
    plt.figure(figsize=(20, 20))
    for i, (image, label) in enumerate(dataset.take(25)):
        subplot_image(5, 5, i + 1, image, CIFAR10_CLASS_NAMES[label.numpy()[0]])
    plt.show()


def plot_tuple(anchor, positive, negative):
    plt.figure(figsize=(9, 3))
    subplot_image(1, 3, 1, anchor)
    subplot_image(1, 3, 2, positive)
    subplot_image(1, 3, 3, negative)
    plt.show()


def get_logdir(subfolder):
    return path.join(path.join(path.join(curdir, "../logs"), subfolder), time.strftime("run_%Y_%m_%d-%H_%M_%S"))


def get_modeldir(name):
    return path.join(path.join(curdir, "../models"), name)


def get_datadir(name):
    return path.join(path.join(curdir, "../data"), name)
