import sys
sys.path.append("..")

from src.model.efficientnet import EfficientNetModel, TARGET_SHAPE, BATCH_SIZE
from src.data.imagenette import Imagenette
from src.data.cifar10 import Cifar10
from src.utils.embeddings import project_embeddings, load_weights_of, get_embeddings_of, save_vectors
from src.model.siamese import SiameseModel

dataset = Imagenette(image_size=TARGET_SHAPE, batch_size=BATCH_SIZE, map_fn=EfficientNetModel.preprocess_input)
# dataset = Cifar10(image_size=TARGET_SHAPE, batch_size=BATCH_SIZE, map_fn=EfficientNetModel.preprocess_input)

model = EfficientNetModel()

emb_vectors, emb_labels = get_embeddings_of(model, dataset)
emb_ds = SiameseModel.prepare_dataset(emb_vectors, emb_labels)

for x in [2, 1.5, 1, 0.75]:
    print("Calculating for margin", x)
    for y in [1, 3, 5, 10, 30]:
        print("Calculating for epochs", y)
        siamese = SiameseModel(model, image_vector_dimensions=512, loss_margin=x, fit_epochs=y)
        siamese.compile()
        siamese.fit(emb_ds, num_classes=dataset.num_classes)

        projection_vectors = siamese.projection_model.predict(emb_vectors)
        save_vectors(projection_vectors, emb_labels, dataset.name + '_' + siamese.name + '_vectors')
        project_embeddings(projection_vectors, emb_labels, siamese.name + '_' + dataset.name)
print('Done!')
