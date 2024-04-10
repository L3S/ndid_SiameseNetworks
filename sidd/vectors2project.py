import glob
from sidd.data.cifar10 import Cifar10
from sidd.data.imagenette import Imagenette
from sidd.data.ukbench import UKBench
from sidd.utils.embeddings import project_embeddings, load_vectors
from sidd.utils.common import get_vectorsdir

TARGET_SHAPE = (224, 224)
dataset_img = Imagenette(batch_size=None, image_size=TARGET_SHAPE)
dataset_c10 = Cifar10(batch_size=None, image_size=TARGET_SHAPE)
dataset_ukb = UKBench(batch_size=None, image_size=TARGET_SHAPE)

DATASETS = [dataset_img, dataset_c10, dataset_ukb]
FEATURES = [512]

vectors = get_vectorsdir('.')
for filepath in glob.iglob(str(get_vectorsdir('sift2/')) + '/*.pbz2'):
    vectors_name = filepath[filepath.index('vectors/') + 8:-5]
    project_name = vectors_name[6:]
    print(vectors_name)
    print(project_name)
    values, labels = load_vectors(get_vectorsdir(vectors_name))
    project_embeddings(values, labels, project_name)


# for dataset in DATASETS:
#     for feat in FEATURES:
#         print('Extracting ' + str(feat) + ' features of HSV from ' + dataset.name)
#         emb_vectors, emb_labels = calc_vectors_fn(dataset.get_combined(), extract_hsv, feat)
#         save_vectors(emb_vectors, emb_labels, 'hsv/' + dataset.name + '_hsv_' + str(feat) + '_vectors')
#         project_embeddings(emb_vectors, emb_labels, 'hsv_' + dataset.name)

print('Done')
