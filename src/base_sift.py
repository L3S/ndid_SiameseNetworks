import sys
sys.path.append("..")

from src.utils.sift import extract_sift
from src.data.cifar10 import Cifar10
from src.data.imagenette import Imagenette
from src.utils.embeddings import calc_vectors_fn, save_vectors

TARGET_SHAPE = (227, 227)
FEATURES = [512, 1024, 2048, 4096]

dataset = Imagenette(batch_size=None, image_size=TARGET_SHAPE)
# dataset = Cifar10(batch_size=None, image_size=TARGET_SHAPE)

for feat in FEATURES:
    print('Extracting ' + str(feat) + ' features of SIFT')
    emb_vectors, emb_labels = calc_vectors_fn(dataset.get_combined(), extract_sift, feat)
    save_vectors(emb_vectors, emb_labels, dataset.name + '_sift_' + str(feat) + '_vectors')

print('Done')
