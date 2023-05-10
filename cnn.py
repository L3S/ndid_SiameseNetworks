import time
import logging as log
import tensorflow as tf
from ndid import SimpleParams
from ndid.data.ukbench import UKBench
from ndid.utils.common import get_modeldir, get_datadir
from ndid.utils.embeddings import project_embeddings, calc_vectors, save_vectors, save_embeddings, load_vectors

tf.get_logger().setLevel('INFO')
log.basicConfig(filename="logfile.log", level=log.INFO, format='%(asctime)s %(message)s')

params = SimpleParams.parse()
dataset = params.get_dataset(
    image_size=params.get_model().get_target_shape(),
    map_fn=params.get_model().preprocess_input
)

if params.ukbench:
    ukbench = UKBench(image_size=params.get_model().get_target_shape(), map_fn=params.get_model().preprocess_input)

inference_model_file = get_modeldir('siamese_inference_' + params.basename + '.tf')
if inference_model_file.exists():
    inference_model = tf.keras.models.load_model(inference_model_file)
    print('Inference model loaded.')
else:
    model_basename = params.model + '_' + dataset.name + '_' + params.seed
    print('Inference model does not exist, training...')
    if params.model != 'efficientnet' and params.model != 'vit':
        model = params.get_model(train_size=len(dataset.get_train()))
        model.compile()
        model_file = get_modeldir(model_basename + '.h5')
        if model_file.exists():
            model.load_weights(model_file)
            print('Model weights loaded.')
        else:
            print('Model weights do not exist, training...')
            start = time.time()
            model.fit(dataset.get_train(), validation_data=dataset.get_val())
            log.info('Model %s trained in %ss', model_basename, time.time() - start)
            model.save_weights(model_file)

            if dataset.get_test() is not None:
                print('Model trained, evaluating...')
                model.evaluate(dataset.get_test())

        emb_model = model.get_embedding_model()
    else:
        emb_model = params.get_model()
        print("Model loaded with Imagenet weights.")

    # projection_vectors, projection_labels = calc_vectors(dataset.get_combined(), emb_model)
    # if params.save_vectors:
    #     save_embeddings(projection_vectors, projection_labels, model_basename + '_vectors')
    # if params.project_vectors:
    #     project_embeddings(projection_vectors, projection_labels, model_basename)
    #
    # if params.ukbench:
    #     print('Computing UKBench vectors...')
    #     ukbench_vectors, ukbench_labels = calc_vectors(ukbench.get_combined(), emb_model)
    #     if params.save_vectors:
    #         save_embeddings(ukbench_vectors, ukbench_labels, 'ukbench_' + model_basename + '_vectors')
    #     if params.project_vectors:
    #         project_embeddings(ukbench_vectors, ukbench_labels, 'ukbench_' + model_basename)

    embeddings_file = get_datadir(emb_model.name + '_' + dataset.name + '_' + params.seed)
    if embeddings_file.exists():
        emb_vectors, emb_labels = load_vectors(embeddings_file)
    else:
        print('Calculating embeddings...')
        emb_vectors, emb_labels = calc_vectors(dataset.get_train(), emb_model)
        save_vectors(emb_vectors, emb_labels, embeddings_file)

    emb_ds = params.get_siamese_class().prepare_dataset(emb_vectors, emb_labels)
    siamese = params.get_siamese_class()(embedding_model=emb_model, image_vector_dimensions=params.dimensions,
                                         loss_margin=params.margin, fit_epochs=params.epochs, basename=params.basename)
    siamese.compile(loss=params.get_loss_class())

    print('Training siamese...')
    start = time.time()
    siamese.fit(emb_ds, num_classes=dataset.num_classes)
    log.info('Siamese model %s trained in %ss', params.basename, time.time() - start)
    siamese.inference_model.save(inference_model_file)
    inference_model = siamese.inference_model

start = time.time()
projection_vectors, projection_labels = calc_vectors(dataset.get_combined(), inference_model)
log.info('Siamese embs %s calculated in %ss', inference_model.name, time.time() - start)
if params.save_vectors:
    save_embeddings(projection_vectors, projection_labels, inference_model.name + '_vectors')
if params.project_vectors:
    project_embeddings(projection_vectors, projection_labels, inference_model.name)

if params.ukbench:
    print('Computing UKBench inference vectors...')
    ukbench_vectors, ukbench_labels = calc_vectors(ukbench.get_combined(), inference_model)
    if params.save_vectors:
        save_embeddings(ukbench_vectors, ukbench_labels, 'ukbench_' + inference_model.name + '_vectors')
    if params.project_vectors:
        project_embeddings(ukbench_vectors, ukbench_labels, 'ukbench_' + inference_model.name)

print('Done!\n')
