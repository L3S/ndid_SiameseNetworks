import sys
sys.path.append("..")

from src.utils.sift import extract_sift
from src.data.cifar10 import Cifar10
from src.data.imagenette import Imagenette
from src.data.ukbench import UKBench
from src.utils.embeddings import calc_vectors_fn, save_vectors

TARGET_SHAPE = (227, 227)
dataset_img = Imagenette(batch_size=None, image_size=TARGET_SHAPE)
dataset_c10 = Cifar10(batch_size=None, image_size=TARGET_SHAPE)
dataset_ukb = UKBench(batch_size=None, image_size=TARGET_SHAPE)

DATASETS = [dataset_img, dataset_c10, dataset_ukb]
FEATURES = [512]

for dataset in DATASETS:
    for feat in FEATURES:
        print('Extracting ' + str(feat) + ' features of SIFT ' + dataset.name)
        emb_vectors, emb_labels = calc_vectors_fn(dataset.get_combined(), extract_sift, feat)
        save_vectors(emb_vectors, emb_labels, 'sift/' + dataset.name + '_sift_' + str(feat) + '_vectors')

print('Done')
