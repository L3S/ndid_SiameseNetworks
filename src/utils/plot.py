import matplotlib.pyplot as plt
from src.data.cifar10 import CLASS_NAMES


def subplot_image(nrows, ncols, index, image, title=None):
    ax = plt.subplot(nrows, ncols, index)
    ax.imshow(image)
    if title is not None:
        ax.set_title(title)
    ax.axis('off')


def plot_grid25(dataset):
    plt.figure(figsize=(20, 20))
    for i, (image, label) in enumerate(dataset.take(25)):
        subplot_image(5, 5, i + 1, image, CLASS_NAMES[label.numpy()[0]])
    plt.show()


def plot_tuple(anchor, positive, negative):
    plt.figure(figsize=(9, 3))
    subplot_image(1, 3, 1, anchor)
    subplot_image(1, 3, 2, positive)
    subplot_image(1, 3, 3, negative)
    plt.show()
