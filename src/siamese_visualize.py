import sys
from pathlib import Path

from tqdm import tqdm

sys.path.append("..")

from utils.common import *
from src.data.cifar10 import *
from src.data.embeddings import *
from tensorflow.keras import layers


def write_embeddings_for_tensorboard(image_vectors: list, labels: list, root_dir: Path):
    import csv
    from tensorboard.plugins import projector
    root_dir.mkdir(parents=True, exist_ok=True)
    with (root_dir / 'values.tsv').open('w') as fp:
        writer = csv.writer(fp, delimiter='\t')
        writer.writerows(image_vectors)

    with (root_dir / 'metadata.tsv').open('w') as fp:
        for lbl in labels:
            fp.write(f'{lbl}\n')

    image_vectors = np.asarray(image_vectors)
    embeddings = tf.Variable(image_vectors, name='embeddings')
    CHECKPOINT_FILE = str(root_dir / 'model.ckpt')
    ckpt = tf.train.Checkpoint(embeddings=embeddings)
    ckpt.save(CHECKPOINT_FILE)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embeddings/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    embedding.tensor_path = 'values.tsv'
    projector.visualize_embeddings(root_dir, config)


inference_model = tf.keras.models.load_model(get_modeldir('cifar10_alexnet.tf'), compile=False)

NUM_SAMPLES_TO_DISPLAY = 10000
LOG_DIR = Path('../logs')
LOG_DIR.mkdir(exist_ok=True, parents=True)

embedding_vds = cifar10_complete()
val_ds = (embedding_vds
          .shuffle(500, seed=42)
          .take(NUM_SAMPLES_TO_DISPLAY)
          .map(process_images_couple)
          .batch(batch_size=32, drop_remainder=False)
          .prefetch(tf.data.AUTOTUNE))

# compute embeddings of the images and their labels, store them in a tsv file for visualization
image_vectors = []
labels = []
for feats_batch in tqdm(val_ds):
    ims = feats_batch[0]
    lbls = feats_batch[1].numpy()
    embs = inference_model(ims).numpy()
    image_vectors.extend(embs.tolist())
    labels.extend(lbls.tolist())

write_embeddings_for_tensorboard(image_vectors, labels, LOG_DIR)

## Model hyperparters
EMBEDDING_VECTOR_DIMENSION = 4096
IMAGE_VECTOR_DIMENSIONS = 512

emb_input_1 = layers.Input(EMBEDDING_VECTOR_DIMENSION)
emb_input_2 = layers.Input(EMBEDDING_VECTOR_DIMENSION)

# projection model is the one to use for queries (put in a sequence after the embedding-generator model above)
projection_model = tf.keras.models.Sequential([
    layers.Dense(IMAGE_VECTOR_DIMENSIONS, activation='tanh', input_shape=(EMBEDDING_VECTOR_DIMENSION,))
])

v1 = projection_model(emb_input_1)
v2 = projection_model(emb_input_2)

embeddings, embedding_labels = load_embeddings('cifar10_alexnet_embeddings')
embeddings_ds = tf.data.Dataset.zip((
    tf.data.Dataset.from_tensor_slices(embeddings),
    tf.data.Dataset.from_tensor_slices(embedding_labels)
))
embeddings_ds = embeddings_ds.cache().shuffle(1000).repeat()

# Do the same with some of the training data, just to see if it works with that
ds = embeddings_ds.take(NUM_SAMPLES_TO_DISPLAY).batch(32).prefetch(tf.data.AUTOTUNE)
_image_vectors = []
_labels = []
for feats_batch in tqdm(ds):
    ims, lbls = feats_batch
    ims = ims.numpy()
    lbls = lbls.numpy()
    embs = projection_model(ims).numpy()
    _image_vectors.extend(embs.tolist())
    _labels.extend(lbls.tolist())
write_embeddings_for_tensorboard(embeddings, embedding_labels, LOG_DIR / 'train')

print('done')
