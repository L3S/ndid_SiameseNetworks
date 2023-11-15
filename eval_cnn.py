import time
import logging as log
import tensorflow as tf

from nnfaiss import knn_search, range_search
from sidd import SimpleParams
from sidd.utils.common import get_modeldir, get_datadir
from sidd.utils.embeddings import project_embeddings, calc_vectors, save_vectors, save_embeddings, load_vectors

tf.get_logger().setLevel('INFO')
log.basicConfig(filename="logfile.log", level=log.INFO, format='%(asctime)s %(message)s')

params = SimpleParams.parse()
dataset = params.get_dataset(
    image_size=params.get_model_class().get_target_shape(),
    map_fn=params.get_model_class().preprocess_input
)

evalds = params.get_eval_dataset(
    image_size=params.get_model_class().get_target_shape(),
    map_fn=params.get_model_class().preprocess_input
)

if evalds is None:
    print('No evaluation dataset specified. Exiting.')
    exit(0)

model_basename = params.model + '_' + dataset.name + '_' + params.seed
print('Inference model does not exist, training...')
model = params.get_model(train_size=len(dataset.get_train()), num_classes=dataset.num_classes, weights=params.weights)
model.compile()
model.summary()

if params.weights == 'imagenet':
    print('Alexnet model loaded, skipping training.')
else:
    model_file = get_modeldir(model_basename + '.h5')
    if model_file.exists():
        print('Loading model weights from %s', model_file)
        model.load_weights(model_file)
    else:
        print('Model weights do not exist. Exiting.')
        exit(0)

emb_model = model.get_embedding_model()

print('Computing ' + evalds.name + ' vectors...')
eval_vectors, eval_labels = calc_vectors(evalds.get_combined(), emb_model)

if params.save_vectors:
    save_embeddings(eval_vectors, eval_labels, 'eval_' + evalds.name + '_' +  model_basename + '_vectors')

if params.compute_stats:
    knn_search.compute_and_save(eval_vectors, eval_labels, 'eval_' + evalds.name + '_' +  model_basename + '_knn', True)
    range_search.compute_and_save(eval_vectors, eval_labels, 'eval_' + evalds.name + '_' +  model_basename + '_range', True)

if params.project_vectors:
    project_embeddings(eval_vectors, eval_labels, 'eval_' + evalds.name + '_' +  model_basename)

print('Done!\n')
