import time
import logging as log
import tensorflow as tf

from nnfaiss import knn_search, range_search
from sidd import SimpleParams
from sidd.utils.common import get_modeldir, get_datadir, get_vectorsdir
from sidd.utils.embeddings import project_embeddings, calc_vectors, save_vectors, save_embeddings, load_vectors

tf.get_logger().setLevel('INFO')
log.basicConfig(filename="logfile.log", level=log.INFO, format='%(asctime)s %(message)s')

params = SimpleParams.parse()
inference_model_file = get_modeldir('siamese_inference_' + params.basename + '.tf')
if inference_model_file.exists():
    inference_model = tf.keras.models.load_model(inference_model_file)
    print('Inference model loaded.')
else:
    raise Exception('Inference model not found. Exiting.')

evalds = params.get_eval_dataset(
    image_size=params.get_model_class().get_target_shape(),
    map_fn=params.get_model_class().preprocess_input
)

if evalds is None:
    print('No evaluation dataset specified. Exiting.')
    exit(0)

print('Computing ' + evalds.name + ' vectors...')
eval_vectors, eval_labels = calc_vectors(evalds.get_combined(), inference_model)
if params.save_vectors:
    save_embeddings(eval_vectors, eval_labels, 'eval_' + evalds.name + '_' +  inference_model.name + '_vectors')

if params.compute_stats:
    knn_search.compute_and_save(eval_vectors, eval_labels, 'eval_' + evalds.name + '_' +  inference_model.name + '_knn', True)
    range_search.compute_and_save(eval_vectors, eval_labels, 'eval_' + evalds.name + '_' +  inference_model.name + '_range', True)

if params.project_vectors:
    project_embeddings(eval_vectors, eval_labels, 'eval_' + evalds.name + '_' +  inference_model.name)

print('Done!\n')
