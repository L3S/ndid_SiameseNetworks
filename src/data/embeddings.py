import _pickle as pickle
from src.utils.common import get_datadir


def save_embeddings(embeddings, labels, title='embeddings_labels'):
    data = [embeddings, labels]

    with open(get_datadir(title + '.pkl'), 'wb') as outfile:
        pickle.dump(data, outfile, -1)


def load_embeddings(title='embeddings_labels'):
    with open(get_datadir(title + '.pkl'), 'rb') as infile:
        result = pickle.load(infile)

    return result[0], result[1]
