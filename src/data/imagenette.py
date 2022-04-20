
def load_dataset():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    images = np.concatenate([train_images, test_images])
    labels = np.concatenate([train_labels, test_labels])
    return tf.data.Dataset.from_tensor_slices((images, labels))
