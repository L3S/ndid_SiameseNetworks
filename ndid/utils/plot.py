import cv2
import numpy as np
import matplotlib.pyplot as plt
from ndid.data.cifar10 import CLASS_NAMES


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


def plot_vectors(vectors):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    _pts = np.random.uniform(size=[500, 3], low=-1, high=1)
    _pts = _pts / np.linalg.norm(_pts, axis=-1)[:, None]
    ax.scatter(_pts[:, 0], _pts[:, 1], _pts[:, 2])

    ax.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2])
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.invert_zaxis()

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def plot_hsv(hist_h, hist_s, hist_v):
    plt.figure()
    ax = plt.subplot()
    ax.set_title("HSV Histogram")
    ax.plot(hist_h, label='H')
    ax.plot(hist_s, label='S')
    ax.plot(hist_v, label='V')
    plt.xlabel("Bins")
    plt.ylabel("Percentage of Pixels")
    plt.legend()
    # ax.axis('off')
    plt.show()


def plot_sift(image, keypoints):
    plt.figure(figsize=(20, 20))
    # from smaller image only smaller number of key points can be extracted
    subplot_image(1, 2, 1, image, "Original image")

    img_kp = image.copy()
    cv2.drawKeypoints(img_kp, keypoints, img_kp, color=(255, 0, 0))

    subplot_image(1, 2, 2, img_kp, "Keypoints")
    plt.show()


def visualizeTuple(anchor, positive, negative):
    """Visualize a few triplets from the supplied batches."""

    def show(ax, image):
        ax.imshow(image)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    fig = plt.figure(figsize=(9, 9))

    axs = fig.subplots(3, 3)
    for i in range(3):
        show(axs[i, 0], anchor[0][i])
        show(axs[i, 1], positive[0][i])
        show(axs[i, 2], negative[0][i])
    plt.show()
