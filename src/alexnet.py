import sys
sys.path.append("..")

from src.model.alexnet import AlexNetModel, TARGET_SHAPE
from src.data.simple3 import Simple3
from src.data.imagenette import Imagenette
from src.data.cifar10 import Cifar10
from src.utils.embeddings import project_embeddings, load_weights_of, get_embeddings_of, save_vectors
from src.model.siamese import SiameseModel

dataset = Imagenette(image_size=TARGET_SHAPE, map_fn=AlexNetModel.preprocess_input)
# dataset = Cifar10(image_size=TARGET_SHAPE, map_fn=AlexNetModel.preprocess_input)
# dataset = Simple3(image_size=TARGET_SHAPE, map_fn=AlexNetModel.preprocess_input)

model = AlexNetModel()
model.compile()
load_weights_of(model, dataset)

emb_vectors, emb_labels = get_embeddings_of(model.get_embedding_model(), dataset)
emb_ds = SiameseModel.prepare_dataset(emb_vectors, emb_labels)

MARGIN = 0.5
siamese = SiameseModel(embedding_model=model.get_embedding_model(), image_vector_dimensions=512)
siamese.compile(loss_margin=MARGIN)
siamese.fit(emb_ds, num_classes=dataset.num_classes)

projection_vectors = siamese.projection_model.predict(emb_vectors)
# save_vectors(projection_vectors, emb_labels, dataset.name + '_' + siamese.name + '_vectors')
project_embeddings(projection_vectors, emb_labels, str(MARGIN) + '_' + siamese.name + '_' + dataset.name)
print('Done!')
