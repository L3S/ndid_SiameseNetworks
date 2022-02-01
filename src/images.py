import sys
sys.path.append("..")

from src.utils.hsv import *
from src.utils.sift import *
import tensorflow as tf
from tensorflow.keras import datasets


# Load dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
cifar10_images = np.concatenate([train_images, test_images])
cifar10_labels = np.concatenate([train_labels, test_labels])
cifar10_vds = tf.data.Dataset.from_tensor_slices((cifar10_images, cifar10_labels))

def print_resized(dataset):
    plt.figure(figsize=(20, 20))
    for i, (image, label) in enumerate(dataset.take(3)):
        img_cv2 = cv2.resize(image.numpy(), target_shape)

        # img_tf = tf.image.per_image_standardization(image)
        img_tf = tf.image.resize(image, target_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        img_tf = tf.cast(img_tf, tf.uint8)
        # img_tf = tf.image.convert_image_dtype(img_tf, dtype=tf.uint8, saturate=False)

        subplot_image(3, 3, i * 3 + 1, image.numpy(), "Original image")
        subplot_image(3, 3, i * 3 + 2, img_cv2, "CV2 image")
        subplot_image(3, 3, i * 3 + 3, img_tf.numpy(), "TF image")
    plt.show()

print_resized(cifar10_vds)

# test HSV
# print('test HSV')
# plot_hsv(cifar10_vds)

# print('test SIFT')
# plot_sift(cifar10_vds)

# smaller
# 906, 1692, 1711, 2610, 3259, 3418, 3789, 4277, 4975, 5010, 5255, 5867, 5988, 6406, 7089, 7365, 8072
# 8443, 8998, 9008, 9323, 9664, 9881, 9903, 9985, 10095, 11650, 13043, 13075, 13841, 14698, 15443
# 16004, 16733, 16888, 18948, 19378, 20015, 20233, 20467, 20621, 20696, 20778, 22672, 22804, 22904
# 23252, 23654, 23985, 25236, 25734, 25931, 27596, 27931, 28016, 28300, 28387, 28807, 30029, 31581
# 32024, 32117, 32629, 32861, 33328, 33489, 33589, 34466, 35063, 35202, 35719, 35877, 35985, 36560
# 36777, 37358, 37439, 38224, 38345, 39942, 40389, 40621, 40864, 41454, 41902, 42017, 43593, 44207
# 44226, 44257, 45801, 47091, 47375, 48663, 48690, 48884, 52366, 52622, 52847, 53227, 53248, 53423
# 53429, 53444, 53660, 53759, 53952, 54957, 55164, 55189, 55762, 56549, 56574, 57105, 57171, 58485
# 58572, 58826, 59318, 59970

# bigger
# 6452, 7365, 7811, 9592, 12075, 15443, 16888, 17623, 22576, 23654, 25931, 33862, 35877, 41902, 44226, 45110, 45801, 48884, 53759, 59318

print('done')
