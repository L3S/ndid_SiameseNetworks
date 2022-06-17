import numpy as np

def getLabelStats(labels):
    """

    :param labels: sequence of labels
    :return:
    """
    counts = np.unique(labels, return_counts=True)
    total_labels = len(counts[0])
    images_per_label = np.max(counts[1])
    return {
        "total_labels": total_labels,
        "images_per_label": images_per_label
    }


def getEmbStats(embeddings):
    dim = embeddings.shape[1]
    return {
        "emb_dimension": dim
    }