import sys
sys.path.append("..")

from src.utils.hsv import *
from src.utils.sift import *
import tensorflow as tf
from tensorflow.keras import datasets


# Load dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
cifar10_vds = tf.data.Dataset.from_tensor_slices((np.concatenate([train_images, test_images]), np.concatenate([train_labels, test_labels])))

# test HSV
print('test HSV')
plot_hsv(cifar10_vds)

print('test SIFT')
plot_sift(cifar10_vds)

print('done')
