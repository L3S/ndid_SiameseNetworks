import sys
sys.path.append("..")

from keras.models import load_model
from src.utils.common import get_modeldir
from src import SimpleParams
from src.data.ukbench import UKBench
from src.model.efficientnet import EfficientNetModel, TARGET_SHAPE, BATCH_SIZE
from src.utils.embeddings import project_embeddings, load_weights_of, get_embeddings_of, calc_vectors, save_vectors

params = SimpleParams.parse()
dataset = params.get_dataset(image_size=TARGET_SHAPE, map_fn=EfficientNetModel.preprocess_input)
basename = 'efficientnet_' + dataset.name + '_d' + str(params.dimensions) + '_m' + str(params.margin) + '_s' + str(params.epochs * 100) + '_' + params.loss

inference_model_file = get_modeldir('siamese_inference_' + basename + '.tf')
if inference_model_file.exists():
    inference_model = load_model(inference_model_file)
else:
    print('Inference Model do not exist, training...')
    model = EfficientNetModel()

    projection_vectors, projection_labels = calc_vectors(dataset.get_combined(), model)
    save_vectors(projection_vectors, projection_labels, 'efficientnet_' + dataset.name + '_vectors')
    project_embeddings(projection_vectors, projection_labels, 'efficientnet_' + dataset.name)

    print('Computing UKBench vectors...')
    ukbench = UKBench(image_size=TARGET_SHAPE, map_fn=EfficientNetModel.preprocess_input)
    ukbench_vectors, ukbench_labels = calc_vectors(ukbench.get_combined(), model)
    save_vectors(ukbench_vectors, ukbench_labels, 'ukbench_efficientnet_' + dataset.name + '_vectors')
    project_embeddings(ukbench_vectors, ukbench_labels, 'ukbench_efficientnet_' + dataset.name)

    emb_ds = params.get_siamese_class().prepare_dataset(*get_embeddings_of(model, dataset))
    siamese = params.get_siamese_class()(embedding_model=model, image_vector_dimensions=params.dimensions,
                                         loss_margin=params.margin, fit_epochs=params.epochs, basename=basename)
    siamese.compile(loss=params.get_loss_class())

    print('Training siamese...')
    siamese.fit(emb_ds, num_classes=dataset.num_classes)
    siamese.inference_model.save(inference_model_file)
    inference_model = siamese.inference_model


projection_vectors, projection_labels = calc_vectors(dataset.get_combined(), inference_model)
save_vectors(projection_vectors, projection_labels, inference_model.name + '_vectors')
project_embeddings(projection_vectors, projection_labels, inference_model.name)

print('Computing UKBench vectors...')
ukbench = UKBench(image_size=TARGET_SHAPE, map_fn=EfficientNetModel.preprocess_input)
ukbench_vectors, ukbench_labels = calc_vectors(ukbench.get_combined(), inference_model)
save_vectors(ukbench_vectors, ukbench_labels, 'ukbench_' + inference_model.name + '_vectors')
project_embeddings(ukbench_vectors, ukbench_labels, 'ukbench_' + inference_model.name)
print('Done!\n')
