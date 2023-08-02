import cv2


def sift_features(image, nfeatures=None):
    sift = cv2.SIFT_create(nfeatures)
    # Calculate the keypoint and each point description of the image
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def extract_sift(image, features=512):
    # the result number of features is the number of keypoints * 128
    nfeatures = int(features / 128)
    keypoints, descriptors = sift_features(image, nfeatures)

    if descriptors is None or len(descriptors) < nfeatures:
        return None
    elif len(descriptors) > nfeatures:
        descriptors = descriptors[:nfeatures]
    return descriptors.flatten()
