import sys
sys.path.append("..")

import tensorflow as tf
from src.model.vit import VitModel, TARGET_SHAPE, BATCH_SIZE
from src.data.imagenette import Imagenette
from src.data.cifar10 import Cifar10
from src.utils.embeddings import project_embeddings, load_weights_of, get_embeddings_of, save_vectors
from src.model.siamese import SiameseModel

dataset = Imagenette(image_size=TARGET_SHAPE, batch_size=BATCH_SIZE, map_fn=VitModel.preprocess_input)
# dataset = Cifar10(image_size=TARGET_SHAPE, batch_size=BATCH_SIZE, map_fn=VitModel.preprocess_input)

model = VitModel()
model.compile()
load_weights_of(model, dataset)

emb_vectors, emb_labels = get_embeddings_of(model, dataset)
emb_ds = SiameseModel.prepare_dataset(emb_vectors, emb_labels)

siamese = SiameseModel(embedding_model=model, image_vector_dimensions=512)
siamese.compile(loss_margin=0.05, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
siamese.fit(emb_ds, num_classes=dataset.num_classes)

projection_vectors = siamese.projection_model.predict(emb_vectors)
save_vectors(projection_vectors, emb_labels, dataset.name + '_' + siamese.name + '_vectors')
project_embeddings(projection_vectors, emb_labels, siamese.name + '_' + dataset.name)
print('Done!')
