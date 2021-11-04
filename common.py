import tensorflow as tf
from os import path, curdir
import time
import matplotlib.pyplot as plt

target_shape = (227, 227)
CIFAR10_CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def process_images_couple(image, label):
    return process_images(image), label


def process_images(image):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, target_shape)
    return image


def plot_first5_fig(dataset):
    plt.figure(figsize=(20, 20))
    for i, (image, label) in enumerate(dataset.take(5)):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(image)
        plt.title(CIFAR10_CLASS_NAMES[label.numpy()[0]])
        plt.axis('off')
    plt.show()


def get_logdir(subfolder):
    return path.join(path.join(path.join(curdir, "logs"), subfolder), time.strftime("run_%Y_%m_%d-%H_%M_%S"))


def get_modeldir(name):
    return path.join(path.join(curdir, "models"), name)


def get_datadir(name):
    return path.join(path.join(curdir, "data"), name)
