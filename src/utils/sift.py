import cv2


def sift_features(image, nfeatures=None):
    sift = cv2.SIFT_create(nfeatures)
    # Calculate the keypoint and each point description of the image
    keypoints, features = sift.detectAndCompute(image, None)
    return keypoints, features


def extract_sift(image, features=512):
    # the result number of features is the number of keypoints * 128
    nfeatures = int(features / 128)
    keypoints, features = sift_features(image, nfeatures)

    if features is None or len(features) < nfeatures:
        return None
    elif len(features) > nfeatures:
        features = features[:nfeatures]
    return features.flatten()
