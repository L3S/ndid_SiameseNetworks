import numpy as np
import cv2

from src.utils.plot import *


def extract_hsv(image, bin0=256, bin1=256, bin2=256):
    """Extract a 3 color channels histogram from the HSV"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # The ranges of the 3 HSV channels in opencv are 0-180, 0-256, 0-256 respectively
    histh = cv2.calcHist([hsv], [0], None, [bin0], [0, 180])
    hists = cv2.calcHist([hsv], [1], None, [bin1], [0, 256])
    histv = cv2.calcHist([hsv], [2], None, [bin2], [0, 256])
    # normalize the histogram
    histh /= histh.sum()
    hists /= hists.sum()
    histv /= histv.sum()
    hist_array = np.append(histh, hists)
    hist_array = np.append(hist_array, histv)
    # return the flattened histogram as the feature vector
    return histh, hists, histv, hist_array.flatten()


def plot_hsv(dataset):
    plt.figure(figsize=(20, 20))
    for i, (image, label) in enumerate(dataset.take(3)):
        # from smaller image only smaller number of key points can be extracted
        img = cv2.resize(image.numpy(), target_shape)

        subplot_image(3, 2, i * 2 + 1, img, "Original image")

        hist0_s, hist1_s, hist2_s, hist_s = extract_hsv(img)
        # print('the length of histogram of the sample', len(hist_s))

        # subplot_image(3, 2, i * 2 + 2, image, "HSV Histogram")
        ax = plt.subplot(3, 2, i * 2 + 2)
        # ax.imshow(image)
        ax.set_title("HSV Histogram")
        ax.plot(hist0_s, label='H')
        ax.plot(hist1_s, label='S')
        ax.plot(hist2_s, label='V')
        plt.xlabel("Bins")
        plt.ylabel("percentage of Pixels")
        plt.legend()
        # ax.axis('off')
    plt.show()
