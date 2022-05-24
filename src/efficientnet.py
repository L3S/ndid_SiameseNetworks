import sys
sys.path.append("..")

from src.data.cifar10 import load_dataset3, NUM_CLASSES
from src.utils.embeddings import project_embeddings, calc_vectors, save_embeddings
from src.utils.common import get_modeldir, get_datadir
from src.model.efficientnet import EfficientNetModel, TARGET_SHAPE, BATCH_SIZE
from src.model.siamese import SiameseModel

model_name = 'cifar10_efficientnet'
embeddings_name = model_name + '_embeddings'

train_ds, val_ds, test_ds = load_dataset3(image_size=TARGET_SHAPE, batch_size=BATCH_SIZE, preprocess_fn=EfficientNetModel.preprocess_input)
comb_ds = train_ds.concatenate(val_ds).concatenate(test_ds)

model = EfficientNetModel()
model.summary()

print('calculating embeddings...')
emb_vectors, emb_labels = calc_vectors(comb_ds, model)
save_embeddings(emb_vectors, emb_labels, embeddings_name)

# emb_vectors, emb_labels = load_embeddings(embeddings_name)

# siamese is the model we train
siamese = SiameseModel(embedding_vector_dimension=1280, image_vector_dimensions=128)
siamese.compile(loss_margin=0.05)
siamese.summary()

ds = SiameseModel.prepare_dataset(emb_vectors, emb_labels)
history = siamese.fit(ds, class_weight={0: 1 / NUM_CLASSES, 1: (NUM_CLASSES - 1) / NUM_CLASSES})

# Build full inference model (from image to image vector):
inference_model = siamese.get_inference_model(model)
inference_model.save(get_modeldir(model_name + '_inference.tf'), save_format='tf', include_optimizer=False)

# inference_model = tf.keras.models.load_model(get_modeldir(model_name + '_inference.tf'), compile=False)

print('visualization')
# compute vectors of the images and their labels, store them in a tsv file for visualization
projection_vectors = siamese.get_projection_model().predict(emb_vectors)
project_embeddings(projection_vectors, emb_labels, model_name + '_siamese')
