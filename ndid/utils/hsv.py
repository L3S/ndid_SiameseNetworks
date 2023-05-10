import cv2
import numpy as np


def hsv_histogram(image, bin_h=256, bin_s=256, bin_v=256):
    """Extract a 3 color channels histogram from the HSV"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # The ranges of the 3 HSV channels in opencv are 0-180, 0-256, 0-256 respectively
    hist_h = cv2.calcHist([hsv], [0], None, [bin_h], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bin_s], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bin_v], [0, 256])
    # normalize the histogram
    hist_h /= hist_h.sum()
    hist_s /= hist_s.sum()
    hist_v /= hist_v.sum()
    return hist_h, hist_s, hist_v


def extract_hsv(image, features=512):
    bin_size = int(features / 3)
    bin_h = bin_size
    bin_s = bin_size
    bin_v = bin_size

    remainder = features % 3
    if remainder == 2:
        bin_s += 1
        bin_v += 1
    elif remainder == 1:
        bin_h += 1

    hist_h, hist_s, hist_v = hsv_histogram(image, bin_h, bin_s, bin_v)

    # return the flattened histogram as the feature vector
    hist_array = np.append(hist_h, hist_s)
    hist_array = np.append(hist_array, hist_v)
    return hist_array.flatten()
