# -*- coding: utf-8 -*-
import numpy as np
import faiss
import time
import collections
import bz2
import pickle
import pandas as pd
import os


def load_embeddings_pkl(name='embeddings'):
    with open(name, 'rb') as infile:
        result = pickle.load(infile)
    return np.array(result[0], dtype='float32'), np.array(result[1])


def load_embeddings_pbz2(name='embeddings', method=None):
    with bz2.BZ2File(name, 'rb') as infile:
        result = pickle.load(infile)
    if method == "SIFT":
        return np.array(result[0], dtype='float32'), np.array(result[1]), np.array(result[2])
    else:
        return np.array(result[0], dtype='float32'), np.array(result[1])


def createIndex(data, indexstring, metric=faiss.METRIC_L2, efConstruction=-1):
    """
    creates from the data (np.array) the index specified in the indexstring
    (string), if the index is not trained, it will be trained on the data given,
    if an HNSW index is used, you can set the efConstruction value. If the
    default value (-1) is kept, the efConstruction is not set (good if not using
    HNSW)
    returns the trained index, the time to build the index and the memory consumed

    """
    starttime = time.perf_counter()

    d = len(data[0])  # dimensionality of the embedding vectors
    index = faiss.index_factory(d, indexstring, metric)  # creation of the index
    if not index.is_trained:
        print("Index is not trained, training now")
        index.train(data)
        print("Index is now trained")
    if efConstruction != -1:
        index.hnsw.efConstruction = efConstruction
    index.add(data)
    stoptime = time.perf_counter()
    print("Building the index took {:.2f}s".format(stoptime - starttime))
    return (index, stoptime - starttime)


def search(data, index, nn, probes=-1):
    """
    given the search query (data, np.array) it searches the index. If the index
    is an IVF index, it visits probes number of cells. Not available for HNSW
    index.
    it returns the distance table D and the ID table I for all vectors which
    have a smaller distance than the threshold to the query vectors.
    for i the ID of the queryvector, the distances and IDs of the corresponding
    indices are stored in X[lim[i]:lim[i+1]]

    """
    starttime = time.perf_counter()
    if probes != -1:
        index.nprobe = probes
    D, I = index.search(data, nn)
    stoptime = time.perf_counter()
    print("Searching took {:.2f} s".format(stoptime - starttime))
    return D, I, stoptime - starttime


def trueMatches(index_labels, query_labels, NN_table):
    """
    given a set of nearest neighbours in the NNtable, it calculates how many
    can also be found in the truthtable (list of tuples) given. IDIndex and Query
    contain the IDs of the index and query vectors. Those IDs are not included
    when creating the index or searching, so the results have to be mapped to
    the original index to use the truthtable.
    """
    nCandidates = 0
    classified = collections.Counter()
    for query_id, NNs in enumerate(NN_table):
        for index_id in NNs:

            if index_labels[index_id] == query_labels[query_id]:
                nCandidates += 1
                classified[(index_labels[index_id], query_labels[query_id])] += 1
            elif index_labels[index_id] < query_labels[query_id]:
                classified[(index_labels[index_id], query_labels[query_id])] += 1
            else:
                classified[(query_labels[query_id], index_labels[index_id])] += 1

    return nCandidates, classified


def findNN(index_data, query_data, index_labels, query_labels, indexstring, treshold, allCandidates, data_table,
           normtime, metric=faiss.METRIC_L2, startstep=1, probe=1, ukbench=False):
    """
    finds (approximately) the number of nearest neigbours, for which the recall is
    above the treshold. The index is build from EmbIndex like specified in
    indexstring, the query is EmbQuery, from the truthtable and IDIndex and
    IDQuery the number of true matches is calculated. In the beginning the
    algorithm increases the number of nearest neighbours by startstep.
    Returns the number of nearest neighbors, the stepsize at the end and the
    recall


    """
    print("starting the search")
    nn = 1
    step = startstep
    recall = 0
    once_above = False
    once_below = False
    print("start value: {}, start step: {}".format(nn, step))
    index, indextime = createIndex(index_data, indexstring, metric)
    D, I, searchtime = search(query_data, index, nn, probe)
    data_table = NN_analysis(index_labels, query_labels, I, data_table, normtime + indextime + searchtime,
                             ukbench=ukbench)
    recall = data_table[-1][5]
    print("NN: {}; Recall: {:.2f}".format(nn, recall))
    while True:
        if (recall >= treshold and (step == 1 or nn == 1)):
            break
        if recall > treshold:
            once_above = True
            if once_below:
                step = max(int(step / 2), 1)
            else:
                step = step * 2
            nn = max(nn - step, 1)
        else:
            once_below = True
            if once_above:
                step = max(int(step / 2), 1)
            else:
                step = step * 2
            nn += step
        D, I, searchtime = search(query_data, index, nn, probe)
        data_table = NN_analysis(index_labels, query_labels, I, data_table, normtime + indextime + searchtime,
                                 ukbench=ukbench)
        recall = data_table[-1][5]
        print("NN: {}; Recall: {:.2f}".format(nn, recall))

    return data_table


def remove_bad_entries(values, labels, vec_size):
    badIDs = []

    for i, vec in enumerate(values):
        if vec.size != vec_size:
            badIDs.append(i)
    values = np.delete(values, badIDs, 0)
    labels = np.delete(labels, badIDs, 0)
    return values, labels


def split_index_query(values, labels, query_size, normed=True):
    if normed:
        values, normtime = norm(values)
    num_labels = len(set(labels))
    query_values = []
    query_labels = []
    query_IDs = []
    collected = collections.Counter()
    i = len(values) - 1
    while sum(collected.values()) < query_size * num_labels:
        if collected[labels[i]] < query_size:
            query_values.append(values[i])
            query_labels.append(labels[i])
            query_IDs.append(i)
            collected[labels[i]] += 1
        i = i - 1
    values = np.delete(values, query_IDs, 0)
    labels = np.delete(labels, query_IDs, 0)
    query_values = np.array(query_values, dtype='float32')
    query_labels = np.array(query_labels)
    return values, labels, query_values, query_labels


def split_index_query_last(values, labels, ratio, normed=True):
    if normed:
        values, normtime = norm(values)
    query_values = []
    query_labels = []
    query_IDs = []
    collected = collections.Counter()
    i = len(values) - 1
    while sum(collected.values()) < ratio * len(values):
        query_values.append(values[i])
        query_labels.append(labels[i])
        query_IDs.append(i)
        collected[labels[i]] += 1
        i = i - 1
    values = np.delete(values, query_IDs, 0)
    labels = np.delete(labels, query_IDs, 0)
    query_values = np.array(query_values, dtype='float32')
    query_labels = np.array(query_labels)
    return values, labels, query_values, query_labels


def split_index_query_ukbench(values, labels, normed=True):
    if normed:
        values, normtime = norm(values)
    query_values = []
    query_labels = []
    query_IDs = []
    index_labels = []
    collected = collections.Counter()
    for i, value in enumerate(values):
        if collected[labels[i] // 4] == 1:
            index_labels.append(labels[i])
        else:
            query_values.append(value)
            query_labels.append(labels[i])
            query_IDs.append(i)
            collected[labels[i] // 4] += 1
    values = np.delete(values, query_IDs, 0)
    # labels = np.delete(labels, query_IDs,0)
    query_values = np.array(query_values, dtype='float32')
    query_labels = np.array(query_labels)
    index_labels = np.array(index_labels)
    return values, index_labels, query_values, query_labels


def norm(values):
    temp = time.perf_counter()
    values = values / np.linalg.norm(values, axis=1).reshape(-1, 1)
    normtime = time.perf_counter() - temp
    print('Values normed!, Norm: ', np.amax(np.linalg.norm(values, axis=1)), np.amin(np.linalg.norm(values, axis=1)))
    return values, normtime


def NN_analysis(index_labels, query_labels, NN_table, data_table, runtime, ukbench=False):
    num_queries, NN = NN_table.shape
    index_labels_u = index_labels.copy()
    query_labels_u = query_labels.copy()
    if ukbench:
        index_labels_u = index_labels_u // 4
        query_labels_u = query_labels_u // 4
    index_label_counts = collections.Counter(index_labels_u)
    query_label_counts = collections.Counter(query_labels_u)
    true_matches = collections.Counter()
    for label in list(index_label_counts):
        true_matches[label] += index_label_counts[label] * query_label_counts[label]
    MAP = 0
    NDCG = 0
    tp_all = 0
    max_matches = sum(true_matches.values())

    for query_id, NNs in enumerate(NN_table):
        tp = 0
        AP = 0
        DCG = 0
        for position, index_id in enumerate(NNs):
            if position >= len(index_labels):
                if tp != index_label_counts[query_labels_u[query_id]]:
                    print('something is wrong here')
                break
            if index_labels_u[index_id] == query_labels_u[query_id]:
                tp += 1
                AP += tp * 1. / (position + 1)
                DCG += 1. / (np.log2(position + 2))

        tp_all += tp
        if tp_all > max_matches:
            print('something went wrong here', query_id, len(NN_table), tp_all, all_matches)

        MAP += AP * 1. / index_label_counts[query_labels_u[query_id]]
        norm = np.sum(1. / np.log2(np.arange(2, tp + 3)))
        NDCG += DCG / norm
    MAP = MAP * 1. / num_queries
    NDCG = NDCG * 1. / num_queries
    Jaccard = tp_all * 1. / (sum(true_matches.values()) - tp_all + num_queries * NN)
    Precision = tp_all / (num_queries * NN)
    Recall = tp_all / max_matches
    f1 = 2 * Precision * Recall / (Precision + Recall)
    Accelleration = len(index_labels) * 1. / NN
    data_table.append([NN, tp_all, max_matches, len(index_labels), num_queries, Recall,
                       Precision, MAP, NDCG, Jaccard, Accelleration, f1, runtime])

    return data_table


def get_file_names(read_all, input_folder, loops={}, analysis='params', skip='', skip_not='', no_siamese=False):
    input_files = []
    output_files = []
    datasets = []
    if read_all:
        print('Scanning now input folder ', input_folder)
        for exp_file in os.scandir(input_folder):
            if exp_file.is_dir():
                continue
            filename_parts = exp_file.name.split('_')

            if skip in filename_parts:
                continue
            elif skip_not != '' and skip_not not in filename_parts:
                continue

            if 'ukbench' in filename_parts:
                dataset = 'ukbench'
            elif 'imagenette' in filename_parts:
                dataset = 'imagenette'
            elif 'cifar10' in filename_parts:
                dataset = 'cifar10'
            else:
                print('WARNING: No dataset could be infered from the filename')
                dataset = None

            if analysis == 'stats':
                if filename_parts[-1] == 'vectors.pbz2':
                    continue
                else:
                    input_files.append(exp_file.name)
                    filename_parts = exp_file.name.split('.')
                    outfile_name = '.'.join(filename_parts[:-1])
                    output_files.append(outfile_name)
                    datasets.append(dataset)

            else:
                if filename_parts[-1] != 'vectors.pbz2':
                    continue
                input_files.append(exp_file.name)
                outfile_name = '_'.join(filename_parts[:-1])
                output_files.append(outfile_name)
                datasets.append(dataset)

    else:
        for model in loops['model']:
            for dataset in loops['dataset']:
                if no_siamese:
                    if dataset == 'ukbench':
                        for dict_name in loops['dict']:
                            infile_name = dataset + '_' + model + '_' + dict_name + '_vectors.pbz2'
                            outfile_name = dataset + '_' + model + '_' + dict_name
                    else:
                        infile_name = model + '_' + dataset + '_vectors.pbz2'
                        outfile_name = model + '_' + dataset
                    output_files.append(outfile_name)
                    input_files.append(infile_name)
                    datasets.append(dataset)

                if model == 'SIFT':
                    if dataset == 'ukbench':
                        for dict_name in loops['dict']:
                            infile_name = dataset + '_siftBOF_dict_' + dict_name + '_d512_vectors'
                            outfile_name = infile_name + '_' + indexstring + '.csv'
                            output_files.append(outfile_name)
                            input_files.append(infile_name)
                            datasets.append(dataset)
                    else:
                        infile_name = dataset + '_siftBOF_dict_' + dataset + '_d512_vectors'
                        outfile_name = infile_name + '_' + indexstring + '.csv'
                        output_files.append(outfile_name)
                        input_files.append(infile_name)
                        datasets.append(dataset)

                elif model == 'hsv':
                    infile_name = dataset + '_hsv_512_vectors.pbz2'
                    outfile_name = dataset + '_hsv'
                    output_files.append(outfile_name)
                    input_files.append(infile_name)
                    datasets.append(dataset)

                else:
                    for s in loops['s']:
                        for m in loops['m']:
                            for l in loops['l']:
                                if analysis == 'params':
                                    if model in ['efficientnet', 'vit']:
                                        filename = dataset + '_' + model + '_siamese_d512_m' + str(m) + '_s' + str(
                                            s) + '_' + l
                                    else:
                                        filename = dataset + '_' + model + '_emb_siamese_d512_m' + str(m) + '_s' + str(
                                            s) + '_' + l

                                    outfile_name = filename
                                    infile_name = filename + '_vectors.pbz2'
                                else:
                                    filename = 'siamese_inference_' + model + '_' + dataset + '_d512_m' + str(
                                        m) + '_s' + str(s) + '_' + l
                                    outfile_name = filename
                                    infile_name = filename + '_vectors.pbz2'

                                output_files.append(outfile_name)
                                input_files.append(infile_name)
                                datasets.append(dataset)
    return input_files, output_files, datasets


if __name__ == '__main__':

    normed = True

    read_all = True
    no_siamese = False
    skip_known = True

    model = 'siamese'
    analysis = 'new_params'

    skip = 'imagenette'
    skip_not = 'alexnet'

    loop_s = [300, 500, 700, 1000, 1500, 2000, 5000]
    loop_m = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    loop_l_large = ['contrastive', 'hard-triplet', 'offline-triplet', 'semi-hard-triplet']
    loop_l_small = ['contrastive', 'offline-triplet']
    loop_model = ['alexnet', 'efficientnet', 'vit', 'vgg16', 'resnet', 'mobilenet']
    loop_data = ['imagenette', 'cifar10', 'ukbench']

    loops = {'s': [500], 'm': [2.0], 'l': ['contrastive'], 'model': ['vgg16'],
             'dataset': ['cifar10'], 'dict': loop_data}

    add_dataset = False
    metric = faiss.METRIC_L2
    normtime = 0
    indexstring = 'Flat'

    output_folder = "/home/schoger/siamese/faiss_analysis/files/"

    if model == 'hsv':
        input_folder = "/home/astappiev/nsir/backup/vectors/hsv/"
        output_folder += model
    elif model == 'SIFT':
        input_folder = "/home/schoger/siamese/SIFT/"
        output_folder += model
    elif analysis == 'stats':
        input_folder = "/home/astappiev/nsir/backup/vectors/stats/"
        output_folder += analysis
    elif analysis == 'params':
        input_folder = "/home/astappiev/nsir/oldrun/vectors/"
        output_folder += analysis
    elif analysis == 'new_params':
        input_folder = "/home/astappiev/nsir/vectors/"
        output_folder += analysis
    else:
        input_folder = "/home/astappiev/nsir/backup/vectors/"
        add_dataset = True

    input_files, output_files, datasets = get_file_names(read_all, input_folder, loops, analysis, skip, skip_not,
                                                         no_siamese)

    num_files = len(input_files)
    full_runtime = 0
    runtime_per_file = 0
    for i, filename in enumerate(input_files):
        if add_dataset:
            fin_output_folder = output_folder + datasets[i] + '/'
        else:
            fin_output_folder = output_folder + '/'

        output_filename = output_files[i] + '_' + indexstring + '.csv'
        if skip_known:
            if output_filename in os.listdir(fin_output_folder):
                print('skipping: ', output_filename)
                continue
        print('Now analysing file:', filename)
        if i > 0:
            print('Last file run {:.2f}'.format(runtime_per_file))
            print('Estimated left runtime: {:.2f}'.format(full_runtime / i * (num_files - i)))

        runtime_per_file = time.perf_counter()

        if datasets[i] == 'ukbench':
            ukbench = True
        else:
            ukbench = False

        if model == 'SIFT':
            index_values, index_names, index_labels = load_embeddings_pbz2(input_folder + filename + '_index.pbz2',
                                                                           method=model)
            query_values, query_names, query_labels = load_embeddings_pbz2(input_folder + filename + '_query.pbz2',
                                                                           method=model)
            if normed:
                index_values, normtime_i = norm(index_values)
                query_values, normtime_q = norm(query_values)
        else:
            values, labels = load_embeddings_pbz2(input_folder + filename)
            if ukbench:
                index_values, index_labels, query_values, query_labels = split_index_query_ukbench(values, labels,
                                                                                                   normed=normed)
            else:
                index_values, index_labels, query_values, query_labels = split_index_query_last(values, labels, 0.2,
                                                                                                normed=normed)

        if ukbench:
            index_counter = collections.Counter(list(index_labels // 4))
            query_counter = collections.Counter(list(query_labels // 4))
        else:
            index_counter = collections.Counter(list(index_labels))
            query_counter = collections.Counter(list(query_labels))

        all_matches = 0
        for label in list(index_counter):
            all_matches += index_counter[label] * query_counter[label]

        data_table = []
        data_table = findNN(index_values, query_values, index_labels, query_labels, indexstring, 0.9, all_matches,
                            data_table, normtime, ukbench=ukbench)

        i_c = len(index_labels)
        i_c_5 = int(i_c / 5)
        m_c = int(sum(index_counter.values()) / len(index_counter.keys()))

        index, indextime = createIndex(index_values, indexstring, metric)
        for nn in [1, 50, 100, 500, i_c_5, 2 * i_c_5, int(i_c / 2), 3 * i_c_5, 4 * i_c_5, i_c, m_c, 2 * m_c]:
            D, I, searchtime = search(query_values, index, nn)
            data_table = NN_analysis(index_labels, query_labels, I, data_table, normtime + indextime + searchtime,
                                     ukbench=ukbench)

        df = pd.DataFrame(data=data_table,
                          columns=['NN', 'true matches', 'GT', 'index_entries', 'query_entries', 'Recall',
                                   'Precision', 'MAP', 'NDCG', 'Jaccard', 'Accelleration', 'F1', 'runtime'])
        print(df)

        df.to_csv(fin_output_folder + output_filename, index=False)

        runtime_per_file = time.perf_counter() - runtime_per_file
        full_runtime += runtime_per_file
