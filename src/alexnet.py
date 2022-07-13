import sys
sys.path.append("..")

from src import SimpleParams
from src.model.alexnet import AlexNetModel, TARGET_SHAPE
from src.utils.embeddings import project_embeddings, load_weights_of, get_embeddings_of, save_vectors

params = SimpleParams.parse()
dataset = params.get_dataset(image_size=TARGET_SHAPE, map_fn=AlexNetModel.preprocess_input)

model = AlexNetModel()
model.compile()
load_weights_of(model, dataset)

emb_model = model.get_embedding_model()
emb_vectors, emb_labels = get_embeddings_of(emb_model, dataset)
emb_ds = params.get_siamese_class().prepare_dataset(emb_vectors, emb_labels)

siamese = params.get_siamese_class()(embedding_model=emb_model, image_vector_dimensions=params.dimensions,
                                     loss_margin=params.margin, fit_epochs=params.epochs)
siamese.compile(loss=params.get_loss_class())
siamese.fit(emb_ds, num_classes=dataset.num_classes)

projection_vectors = siamese.projection_model.predict(emb_vectors)
save_vectors(projection_vectors, emb_labels, dataset.name + '_' + siamese.name + '_' + params.loss + '_vectors')
project_embeddings(projection_vectors, emb_labels, siamese.name + '_' + params.loss + '_' + dataset.name)
print('Done!\n')
