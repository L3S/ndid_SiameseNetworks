import time
import tensorflow as tf
from pathlib import Path

def normalize_image(image):
    image = tf.image.per_image_standardization(image)
    return image


def resize_image(image, target_size):
    return tf.image.resize(image, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def get_project_root():
    return Path(__file__, "../../..").resolve()


def get_dataset(name):
    return Path(get_project_root(), 'datasets', name)


def get_logdir_root():
    return Path(get_project_root(), 'logs')


def get_logdir(subfolder):
    return Path(get_logdir_root(), subfolder, time.strftime("run_%Y_%m_%d-%H_%M_%S"))


def get_modeldir(name):
    return Path(get_project_root(), "models", name)


def get_datadir(name):
    return Path(get_project_root(), "data", name)


def get_vectorsdir(name):
    return Path(get_project_root(), "vectors", name)
