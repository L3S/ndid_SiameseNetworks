import matplotlib.pyplot as plt
import tensorflow as tf


def plot_batch(image_batch, label_batch, grid_size=(4, 4)):
    figures = min(len(image_batch), grid_size[0] * grid_size[1])

    for n in range(figures):
        image = image_batch[n]
        if tf.is_tensor(image):
            image = image.numpy()

        label = label_batch[n]
        if tf.is_tensor(label):
            label = label.numpy()

        ax = plt.subplot(grid_size[0], grid_size[1], n + 1)
        plt.imshow(image, cmap="viridis")
        plt.title(label, fontsize=32)
        plt.axis("off")
    plt.show()


def is_batched(dataset):
    if not isinstance(dataset, tf.data.Dataset):
      raise TypeError('dataset is not a tf.data.Dataset')
    
    if dataset.__class__.__name__ == '_UnbatchDataset':
        return False

    input_dataset = dataset._input_dataset
    while not hasattr(input_dataset, '_batch_size') and hasattr(input_dataset, '_input_dataset'):
      input_dataset = input_dataset._input_dataset

    if hasattr(input_dataset, '_batch_size'):
      return True

    return False


def plot_sample(ds, max_images=16):
    if is_batched(ds):
      ds = ds.take(1).unbatch()

    x_images = []
    x_labels = []

    for image, label in ds:
        x_images.append(image)
        x_labels.append(label)

        if len(x_images) >= max_images:
            break

    plot_batch(x_images, x_labels)


def plot_label(ds, label, max_images=32):
    x_images = []
    x_labels = []

    for image, label in ds.unbatch().filter(lambda i, l: l == label):
        x_images.append(image)
        x_labels.append(label)

        if len(x_images) >= max_images:
            break

    plot_batch(x_images, x_labels)
