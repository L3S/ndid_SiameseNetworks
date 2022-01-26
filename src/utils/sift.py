import matplotlib.pyplot as plt
import cv2

from src.utils.common import *


def extract_sift(image):
    sift = cv2.SIFT_create(8)
    # Calculate the keypoint and each point description of the image
    keypoints, features = sift.detectAndCompute(image, None)
    return keypoints, features


def plot_sift(dataset):
    plt.figure(figsize=(20, 20))
    for i, (image, label) in enumerate(dataset.take(3)):
        # from smaller image only smaller number of key points can be extracted
        img = cv2.resize(image.numpy(), target_shape)

        subplot_image(3, 2, i * 2 + 1, img, "Original image")

        keypoints, features = extract_sift(img)
        img_kp = img.copy()
        cv2.drawKeypoints(img_kp, keypoints, img_kp, color=(255, 0, 0))

        subplot_image(3, 2, i * 2 + 2, img_kp, "Keypoints")
    plt.show()
