# -*- coding: utf-8 -*-
import numpy as np
import faiss
import time
import collections
import bz2
import pickle
import pandas as pd
import os

from ndid.utils.common import get_faissdir


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


def trueMatches(index_labels, query_labels, NN_table, ukbench=False):
    """
    given a set of nearest neighbours in the NNtable, it calculates how many
    can also be found in the truthtable (list of tuples) given. IDIndex and Query
    contain the IDs of the index and query vectors. Those IDs are not included
    when creating the index or searching, so the results have to be mapped to
    the original index to use the truthtable.
    """
    nCandidates = 0
    index_labels_u = np.array(index_labels.copy())
    query_labels_u = np.array(query_labels.copy())
    if ukbench:
        index_labels_u = index_labels_u // 4
        query_labels_u = query_labels_u // 4
    # classified = collections.Counter()
    NN_labels = index_labels_u[NN_table]
    # NN_labels = np.empty_like(NN_table)
    # for row_ID, row in enumerate(NN_table):
    #     NN_labels[row_ID,:] = index_labels_u[row].copy()

    query_labels_u = np.reshape(query_labels_u, (-1, 1))

    truthtable = NN_labels == query_labels_u

    all_matches = query_labels_u == index_labels_u

    allMatch_perQuery = np.reshape(np.sum(all_matches, axis=1), (-1, 1))
    nCandidates = np.sum(truthtable)

    return nCandidates, truthtable, allMatch_perQuery


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
    norm = np.linalg.norm(values, axis=1)
    if (norm == 0).any():
        print('0 Norm!')
        norm[norm == 0] = np.inf
    values = values / norm.reshape(-1, 1)
    normtime = time.perf_counter() - temp
    print('Values normed!, Norm: ', np.amax(np.linalg.norm(values, axis=1)), np.amin(np.linalg.norm(values, axis=1)))
    return values, normtime


def NN_analysis(index_labels, query_labels, NN_table, D_table, data_table, runtime, ukbench=False):
    _, truth_table, matches_query = trueMatches(index_labels, query_labels, NN_table, ukbench=ukbench)
    print('Min number of matches per Query: ', np.amin(matches_query))
    num_queries, NN = truth_table.shape
    D_table = np.sqrt(D_table)
    distances = np.quantile(D_table, np.linspace(0.0, 1, num=101))
    if distances.max() < 2.0:
        distances = np.append(distances, 2.0)

    # distances = np.reshape(distances, (-1, 1, 1))

    # lower_dist = D_table < distances
    # print(lower_dist.shape)
    # t_included = truth_table * lower_dist
    # k_included = np.cumsum(lower_dist, axis=2, dtype=float)

    # k_included[k_included == 0] = np.inf

    # candidates = np.cumsum(t_included, axis=2)
    # Recall = candidates/matches_query
    # if (Recall>1).any():
    #     print(np.amax(Recall))
    #     raise AssertionError ('Recall > 1, please check!')
    # Precision = candidates/k_included
    # Jaccard = candidates/(k_included+matches_query-candidates)
    # F1 = 2*Precision*Recall/(Precision+Recall)
    # F1[np.isnan(F1)]=0
    # Accelleration = len(index_labels)/k_included

    # Recall = np.mean(Recall, axis=1)
    # Precision = np.mean(Precision, axis=1)
    # F1 = np.mean(F1, axis=1)
    # Jaccard = np.mean(Jaccard, axis=1)
    # Accelleration = np.mean(Accelleration, axis=1)

    # AP = np.cumsum(candidates/k_included*t_included,axis=2)*1/matches_query
    # MAP = np.sum(AP, axis=1)/num_queries
    # DCG = np.cumsum(1./(np.log2(k_included+1))*t_included, axis=2)
    # norm_DCG = np.cumsum(1./np.log2(np.arange(2, np.amax(matches_query)+2)))
    # NDCG = np.sum(DCG*1./norm_DCG[candidates-1], axis=1)/num_queries

    # true_matches = np.sum(candidates, axis=1)

    # print(MAP)

    # data = {'Range': distances.flatten(), 'true matches': true_matches[:,-1],
    #         'GT': np.full((100), np.sum(matches_query)),
    #         'index_entries':np.full(100, len(index_labels)),
    #         'query_entries':np.full(100, len(query_labels)),
    #         'Recall':Recall[:,-1], 'Precision':Precision[:,-1],
    #         'MAP':MAP[:,-1], 'NDCG':NDCG[:,-1],
    #         'Jaccard':Jaccard[:,-1], 'Accelleration':Accelleration[:,-1], 'F1':F1[:,-1]}

    data = np.empty((distances.shape[0], 13))
    columns = ['Range', 'candidates', 'true matches',
               'GT', 'index_entries', 'query_entries', 'Recall', 'Precision',
               'MAP', 'NDCG', 'Jaccard', 'Accelleration', 'F1']

    for i, distance in enumerate(distances):
        print('Step ', i, 'Distance ', distance)

        lower_dist = D_table <= distance
        t_included = truth_table * lower_dist
        k_included = np.cumsum(lower_dist, axis=1, dtype=float)
        max_k = int(np.amax(k_included))

        t_included = t_included[:, :max_k]
        k_included = k_included[:, :max_k]

        k_included[k_included == 0] = np.inf

        if np.sum(lower_dist) == 0:
            print(np.sum(lower_dist))
            data[i, :] = [distance, np.sum(lower_dist), 0, 0,
                          0, 0, 0,
                          0, 0, 0, 0,
                          0, 0]
            continue
        candidates = np.cumsum(t_included, axis=1)
        Recall = candidates / matches_query
        if (Recall > 1).any():
            print(np.amax(Recall))
            raise AssertionError('Recall > 1, please check!')
        Precision = candidates / k_included
        Jaccard = candidates / (k_included + matches_query - candidates)
        F1 = 2 * Precision * Recall / (Precision + Recall)
        F1[np.isnan(F1)] = 0
        Accelleration = len(index_labels) / k_included

        Recall = np.mean(Recall, axis=0)
        Precision = np.mean(Precision, axis=0)
        F1 = np.mean(F1, axis=0)
        Jaccard = np.mean(Jaccard, axis=0)
        Accelleration = np.mean(Accelleration, axis=0)

        AP = np.cumsum(candidates / k_included * t_included, axis=1) * 1 / matches_query
        MAP = np.sum(AP, axis=0) / num_queries
        DCG = np.cumsum(1. / (np.log2(k_included + 1)) * t_included, axis=1)
        norm_DCG = np.cumsum(1. / np.log2(np.arange(2, np.amax(matches_query) + 2)))
        NDCG = np.sum(DCG * 1. / norm_DCG[candidates - 1], axis=0) / num_queries

        true_matches = np.sum(candidates, axis=0).flatten()

        data[i, :] = [distance, np.sum(lower_dist), true_matches[-1], np.sum(matches_query),
                      len(index_labels), len(query_labels), Recall[-1],
                      Precision[-1], MAP[-1], NDCG[-1], Jaccard[-1],
                      Accelleration[-1], F1[-1]]

    data_table = pd.DataFrame(data=data, columns=columns)
    return data_table


def get_file_names(read_all, input_folder, loops={}, analysis='params', skip='', skip_not='', no_siamese=False):
    input_files = []
    output_files = []
    datasets = []
    models = []
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

            if filename_parts[0] == 'siamese':
                dataset = filename_parts[3]
                model = filename_parts[2]
            else:
                if filename_parts[0] not in set(['imagenette', 'cifar10', 'ukbench']):
                    model = filename_parts[0]
                    dataset = filename_parts[1]
                else:
                    dataset = filename_parts[0]
                    if dataset == 'ukbench':

                        if filename_parts[1] == 'siamese':
                            model = filename_parts[3]
                        else:
                            model = filename_parts[1]
                    else:
                        model = filename_parts[1]

            if analysis == 'stats':
                if filename_parts[-1] == 'vectors.pbz2':
                    continue
                else:
                    input_files.append(exp_file.name)
                    filename_parts = exp_file.name.split('.')
                    outfile_name = '.'.join(filename_parts[:-1])
                    output_files.append(outfile_name)
                    datasets.append(dataset)
                    models.append(model)

            else:
                if filename_parts[-1] != 'vectors.pbz2':
                    continue
                input_files.append(exp_file.name)
                outfile_name = '_'.join(filename_parts[:-1])
                output_files.append(outfile_name)
                datasets.append(dataset)
                models.append(model)

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
                    models.append(model)

                if model == 'SIFT':
                    if dataset == 'ukbench':
                        for dict_name in loops['dict']:
                            infile_name = dataset + '_siftBOF_dict_' + dict_name + '_d512_vectors'
                            outfile_name = dataset + '_siftBOF_dict_' + dict_name
                            output_files.append(outfile_name)
                            input_files.append(infile_name)
                            datasets.append(dataset)
                            models.append(model)
                    else:
                        infile_name = dataset + '_siftBOF_dict_' + dataset + '_d512_vectors'
                        outfile_name = dataset + '_siftBOF_dict_' + dataset
                        output_files.append(outfile_name)
                        input_files.append(infile_name)
                        datasets.append(dataset)
                        models.append(model)

                elif model == 'hsv':
                    infile_name = dataset + '_hsv_512_vectors.pbz2'
                    outfile_name = dataset + '_hsv'
                    output_files.append(outfile_name)
                    input_files.append(infile_name)
                    datasets.append(dataset)
                    models.append(model)

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
                                    output_files.append(outfile_name)
                                    input_files.append(infile_name)
                                    datasets.append(dataset)
                                    models.append(model)
                                elif analysis == 'new_params':
                                    for date in loops['date']:
                                        if date > 10000:
                                            filename = 'siamese_inference_' + model + '_' + dataset + '_d512_m' + str(
                                                m) + '_s' + str(s) + '_' + l + '_' + str(date)
                                            outfile_name = filename
                                            infile_name = filename + '_vectors.pbz2'

                                            output_files.append(outfile_name)
                                            input_files.append(infile_name)
                                            datasets.append(dataset)
                                            models.append(model)
                                        else:
                                            for num in range(1, 6):
                                                filename = 'siamese_inference_' + model + '_' + dataset + '_d512_m' + str(
                                                    m) + '_s' + str(s) + '_' + l + '_' + str(date) + str(num)
                                                outfile_name = filename
                                                infile_name = filename + '_vectors.pbz2'

                                                output_files.append(outfile_name)
                                                input_files.append(infile_name)
                                                datasets.append(dataset)
                                                models.append(model)

                                else:
                                    if dataset == 'ukbench':
                                        for dict_name in loops['dict']:
                                            filename = dataset + '_siamese_inference_' + model + '_' + dict_name + '_d512_m' + str(
                                                m) + '_s' + str(s) + '_' + l
                                            outfile_name = filename
                                            infile_name = filename + '_vectors.pbz2'

                                            output_files.append(outfile_name)
                                            input_files.append(infile_name)
                                            datasets.append(dataset)
                                            models.append(model)
                                    else:
                                        filename = 'siamese_inference_' + model + '_' + dataset + '_d512_m' + str(
                                            m) + '_s' + str(s) + '_' + l
                                        outfile_name = filename
                                        infile_name = filename + '_vectors.pbz2'

                                        output_files.append(outfile_name)
                                        input_files.append(infile_name)
                                        datasets.append(dataset)
                                        models.append(model)
    return input_files, output_files, datasets, models


def compute_and_save(values, labels, name='embeddings', ukbench=False):
    if ukbench:
        index_values, index_labels, query_values, query_labels = split_index_query_ukbench(values, labels)
    else:
        index_values, index_labels, query_values, query_labels = split_index_query_last(values, labels, 0.2)

    index, indextime = createIndex(index_values, 'Flat', faiss.METRIC_L2)
    D, I, searchtime = search(query_values, index, len(index_labels))
    data_table = NN_analysis(index_labels, query_labels, I, D, [], indextime + searchtime, ukbench=ukbench)
    data_table.to_csv(str(get_faissdir(name)) + '.csv', index=False)

    if ukbench:
        I_labels = index_labels[I]
        f = open(str(get_faissdir(name)) + '_IDs.txt', 'w')
        for i in range(10):
            print(*[query_labels[i], I_labels[i,:5]], sep = '\t', file=f)


if __name__ == '__main__':

    normed = True

    read_all = True
    no_siamese = True
    skip_known = True

    model = 'siamese'
    analysis = 'new_params'

    skip = ''
    skip_not = 'cifar10'

    loop_s = [300, 500, 700, 1000, 1500, 2000, 5000]
    loop_m = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    loop_l_large = ['contrastive', 'hard-triplet', 'offline-triplet', 'semi-hard-triplet']
    loop_l_small = ['contrastive', 'offline-triplet']
    loop_model = ['alexnet', 'efficientnet', 'vit', 'vgg16', 'resnet', 'mobilenet']
    loop_data = ['imagenette', 'cifar10', 'ukbench']

    loops = {'s': [500], 'm': [1.0], 'l': ['contrastive'], 'model': loop_model,
             'dataset': ['ukbench', 'imagenette', 'cifar10'], 'dict': ['imagenette', 'cifar10']}

    add_dataset = False
    add_model = False
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
        add_model = True
    else:
        input_folder = "/home/astappiev/nsir/backup/vectors/"
        add_dataset = True

    print('Fetching all files...')

    input_files, output_files, datasets, models = get_file_names(read_all, input_folder, loops, analysis, skip,
                                                                 skip_not, no_siamese)

    num_files = len(input_files)
    print('Number of files to analyse: ', num_files)
    full_runtime = 0
    runtime_per_file = 0
    file_counter = 0
    for i, filename in enumerate(input_files):
        if add_dataset:
            fin_output_folder = output_folder + datasets[i] + '/'
        elif add_model:
            fin_output_folder = output_folder + '/' + models[i] + '/'
        else:
            fin_output_folder = output_folder + '/'

        print(fin_output_folder, filename)
        output_filename = output_files[i] + '_' + 'Range_new' + '.csv'
        if skip_known:
            if output_filename in os.listdir(fin_output_folder):
                print('skipping: ', output_filename)
                continue
        print('Now analysing file:', filename)

        print('File ' + str(i + 1) + '/' + str(num_files))
        if file_counter > 0:
            print('Last file run {:.2f}'.format(runtime_per_file))
            print('Estimated left runtime: {:.2f}'.format(full_runtime / file_counter * (num_files - i)))

        file_counter += 1
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
        # data_table = findNN(index_values, query_values, index_labels, query_labels, indexstring, 0.9, all_matches, data_table, normtime, ukbench=ukbench)

        i_c = len(index_labels)
        i_c_5 = int(i_c / 5)
        m_c = int(sum(index_counter.values()) / len(index_counter.keys()))

        index, indextime = createIndex(index_values, indexstring, metric)
        for nn in [i_c]:
            D, I, searchtime = search(query_values, index, nn)
            data_table = NN_analysis(index_labels, query_labels, I, D, data_table, normtime + indextime + searchtime,
                                     ukbench=ukbench)

        data_table.to_csv(fin_output_folder + output_filename, index=False)

        runtime_per_file = time.perf_counter() - runtime_per_file
        full_runtime += runtime_per_file
