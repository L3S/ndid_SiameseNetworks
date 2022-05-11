import sys
sys.path.append("..")

from src.utils.hsv import *
from src.utils.sift import *
from src.utils.embeddings import *
from src.data.cifar10 import *

train_ds, test_ds = load_dataset()
cifar10_vds = train_ds.concatenate(test_ds)


def export_hsv(bin0=256, bin1=256, bin2=256):
    header = ['ID', 'Label', 'HSV vector']
    with open('../data/hsv_' + str(bin0) + '.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(header)

        for i, (image, label) in enumerate(cifar10_vds):
            a, b, c, hist_array = extract_hsv(image.numpy(), bin0, bin1, bin2)
            label_str = ','.join(map(str, label.numpy()))
            value_str = ','.join(map(str, hist_array))
            writer.writerow([i, label_str, value_str])


def export_sift(nfeatures=8):
    header = ['ID', 'Label', 'SIFT descriptors']
    with open('../data/sift_' + str(nfeatures) + '.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter=";")
        # write the header
        writer.writerow(header)

        for i, (image, label) in enumerate(cifar10_vds):
            keypoints, features = extract_sift(image.numpy(), nfeatures)
            label_str = ','.join(map(str, label.numpy()))
            if features is not None:
                value_str = ','.join(map(str, features.flatten()))
            else:
                value_str = 'None'
                print('Unable to extract SIFT from image', i)
            writer.writerow([i, label_str, value_str])


def export_embeddings():
    embeddings, embeddings_labels = load_embeddings(name='cifar10_alexnet_embeddings')
    export_embeddings(embeddings, embeddings_labels, 'alexnet')


# HSV
# export_hsv(170, 171, 171) # 512
# export_hsv(340, 342, 342) # 1024
# export_hsv(682, 683, 683) # 2048
# export_hsv(1366, 1365, 1365) # 4096

# SIFT
# export_sift(4)
# export_sift(8)
# export_sift(16)
# export_sift(32)

# Siamese Embeddings
export_embeddings()
print('done')
