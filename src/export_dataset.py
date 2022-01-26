import sys
sys.path.append("..")

import csv
from src.utils.hsv import *
from src.utils.sift import *
import tensorflow as tf
from utils.common import *
from utils.distance import *
from src.data.embeddings import *
from src.model.alexnet import AlexNetModel
from tensorflow.keras import layers, Model, models, datasets

# Load dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
cifar10_images = np.concatenate([train_images, test_images])
cifar10_labels = np.concatenate([train_labels, test_labels])
cifar10_vds = tf.data.Dataset.from_tensor_slices((cifar10_images, cifar10_labels))

def export_hsv():
    header = ['ID', 'Label', 'HSV vector']
    with open('../data/hsv.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=";")
        # write the header
        writer.writerow(header)

        for i, (image, label) in enumerate(cifar10_vds):
            a, b, c, hist_array = extract_hsv(image.numpy())
            label_str = ','.join(map(str, label.numpy()))
            value_str = ','.join(map(str, hist_array))
            writer.writerow([i, label_str, value_str])


def export_sift():
    header = ['ID', 'Label', 'SIFT descriptors']
    with open('../data/sift.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=";")
        # write the header
        writer.writerow(header)

        for i, (image, label) in enumerate(cifar10_vds):
            # from smaller image only smaller number of key points can be extracted
            img = cv2.resize(image.numpy(), (230, 230))
            keypoints, features = extract_sift(img)
            label_str = ','.join(map(str, label.numpy()))
            if features is not None:
                value_str = ','.join(map(str, features.flatten()))
            else:
                value_str = 'None'
                print('Unable to extract SIFT from image', i)
            writer.writerow([i, label_str, value_str])


def export_embeddings():
    header = ['ID', 'Label', 'Siamese Embeddings']
    with open('../data/siamese.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=";")
        # write the header
        writer.writerow(header)

        seamese = models.load_model(get_modeldir('seamese_cifar10.tf'))

        embedding_vds = (cifar10_vds.map(process_images_couple).batch(batch_size=32, drop_remainder=False))
        print('predicting embeddings')
        embeddings = seamese.predict(embedding_vds)
        print('embeddings done')

        for i, (label) in enumerate(cifar10_labels):
            label_str = ','.join(map(str, label))
            value_str = ','.join(map(str, embeddings[i]))
            writer.writerow([i, label_str, value_str])


# export_hsv()
# export_sift()
export_embeddings()
print('done')
