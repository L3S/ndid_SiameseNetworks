import sys
sys.path.append("..")

import tensorflow as tf
from src.model.mobilenet import MobileNetModel, PRETRAIN_EPOCHS, TARGET_SHAPE
from src.data.imagenette import Imagenette
from src.data.cifar10 import Cifar10
from src.utils.embeddings import project_embeddings, load_weights_of, get_embeddings_of, save_vectors
from src.model.siamese import SiameseModel

dataset = Imagenette(image_size=TARGET_SHAPE, map_fn=MobileNetModel.preprocess_input)
# dataset = Cifar10(image_size=TARGET_SHAPE, map_fn=MobileNetModel.preprocess_input)
PRETRAIN_TOTAL_STEPS = PRETRAIN_EPOCHS * len(dataset.get_train())

model = MobileNetModel()
model.compile(optimizer=tf.keras.optimizers.RMSprop(tf.keras.optimizers.schedules.CosineDecay(1e-3, PRETRAIN_TOTAL_STEPS)))
load_weights_of(model, dataset)

emb_model = model.get_embedding_model()
emb_vectors, emb_labels = get_embeddings_of(emb_model, dataset)
emb_ds = SiameseModel.prepare_dataset(emb_vectors, emb_labels)

for x in [2, 1.5, 1, 0.75]:
    print("Calculating for margin", x)
    for y in [1, 3, 5, 10, 30]:
        print("Calculating for epochs", y)
        siamese = SiameseModel(emb_model, image_vector_dimensions=512, loss_margin=x, fit_epochs=y)
        siamese.compile()
        siamese.fit(emb_ds, num_classes=dataset.num_classes)

        projection_vectors = siamese.projection_model.predict(emb_vectors)
        save_vectors(projection_vectors, emb_labels, dataset.name + '_' + siamese.name + '_vectors')
        project_embeddings(projection_vectors, emb_labels, siamese.name + '_' + dataset.name)
print('Done!')
