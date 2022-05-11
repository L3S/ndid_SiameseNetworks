import tensorflow as tf
from os import path, curdir
import time


def normalize_image(image):
    image = tf.image.per_image_standardization(image)
    return image


def resize_image(image, target_size):
    return tf.image.resize(image, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def get_logdir_root():
    return path.join(curdir, '../logs')


def get_logdir(subfolder):
    return path.join(path.join(get_logdir_root(), subfolder), time.strftime("run_%Y_%m_%d-%H_%M_%S"))


def get_modeldir(name):
    return path.join(path.join(curdir, "../models"), name)


def get_datadir(name):
    return path.join(path.join(curdir, "../data"), name)
