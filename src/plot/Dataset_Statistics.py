# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 13:34:14 2021

@author: user
"""

import numpy as np
import faiss
import time
import psutil
import collections
import matplotlib.pyplot as plt


def createIndex(data, indexstring, metric = faiss.METRIC_L2, efConstruction=-1):
    """
    creates from the data (np.array) the index specified in the indexstring
    (string), if the index is not trained, it will be trained on the data given,
    if an HNSW index is used, you can set the efConstruction value. If the 
    default value (-1) is kept, the efConstruction is not set (good if not using
    HNSW)
    returns the trained index, the time to build the index and the memory consumed

    """
    starttime = time.perf_counter()
    memorystart = psutil.Process().memory_info().rss / (1024 * 1024)
    d = len(data[0])                  #dimensionality of the embedding vectors
    index = faiss.index_factory(d, indexstring, metric)         #creation of the index
    if not index.is_trained:
        print("Index is not trained, training now")
        index.train(data)
    print("Index is now trained")
    if efConstruction!=-1:
        index.hnsw.efConstruction = efConstruction
    index.add(data)
    memorystop = psutil.Process().memory_info().rss / (1024 * 1024)
    stoptime = time.perf_counter()
    print("Building the index took {:.2f}s".format(stoptime-starttime))
    return(index, stoptime-starttime, memorystop-memorystart)

def search(data, index, nNeighbours, probes = 1):
    """
    given the search query (data, np.array) it searches the index. If the index
    is an IVF index, it visits probes number of cells.
    it returns the distance table D and the ID table I for the number of nearest
    neighbours specified in nNeighbours. The number of a row is the ID of the
    query vector, while the entries are the distances to / IDs of the index
    vectors

    """
    starttime = time.perf_counter()
    memorystart = psutil.Process().memory_info().rss / (1024 * 1024)
    index.nprobe = probes
    D,I = index.search(data, nNeighbours)
    memorystop = psutil.Process().memory_info().rss / (1024 * 1024)
    stoptime = time.perf_counter()
    print("Searching took {:.2f} s".format(stoptime-starttime))
    return D,I, stoptime-starttime, memorystop-memorystart



def trueMatches(IndexLabel, QueryLabel, NNtable):
    """
    given a set of nearest neighbours in the NNtable, it calculates how many
    can also be found in the truthtable (list of tuples) given. IDIndex and Query
    contain the IDs of the index and query vectors. Those IDs are not included
    when creating the index or searching, so the results have to be mapped to
    the original index to use the truthtable.
    """
    nCandidates = 0
    for i in range(len(QueryLabel)):
        for candidate in NNtable[i]:
            if IndexLabel[candidate] == QueryLabel[i]:
                nCandidates +=1
    return nCandidates
    

def findNN (Index, Query, IndexLabel, QueryLabel,indexstring, treshold, allCandidates, metric = faiss.METRIC_L2, startstep=1000, probe=1):
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
    step =startstep
    recall = 0
    onceabove = False
    print("start value: {}, start step: {}".format(nn, step))
    index, indextime,indexmemory = createIndex(Index, indexstring, metric)
    D,I,searchtime,searchmemory = search(Query, index, nn, probe)
    matches = trueMatches(IndexLabel, QueryLabel, I)
    recall = matches / allCandidates
    print(recall, nn)
    while ((step > 2 or recall < treshold) and nn>0):
        if recall > treshold:
            onceabove = True
            if nn == 1: break
            step = max(int(step/2),1)
            nn = nn - step
        else:
            if onceabove: step=max(int(step/2),1)
            nn = nn + step
        D,I,searchtime,searchmemory = search(Query, index, nn, probe)
        matches = trueMatches(IndexLabel, QueryLabel, I)
        recall = matches / allCandidates

        print(recall, nn)

    return nn, step, recall



if __name__ == '__main__':
    
    #Main infos for starting
    main_dir = '../Processed/' #folder of the datasets and embeddings
    method = 'sift'                       
    vectorsize = 512
    filename = method + '_{}'.format(vectorsize)

    
    
    metric = faiss.METRIC_INNER_PRODUCT #faiss.METRIC_L2
    normed = True
    normtime = 0
    indexstring = 'Flat'
    
    outstring = 'statistics_'+filename
    
    if metric == faiss.METRIC_L2:
        outring+='_L2'
    else:
        outstring+='_IP'
    if normed:
        outstring += '_normed'
    outfile = open(outstring+".txt", 'w')
    
    for i in range(11):
        if i == 10:
            Index = np.load(main_dir + filename +'_index_Embedding' +'.npy')
            IndexID = np.load(main_dir + filename +'_index_ID' + '.npy')
            IndexLabel = np.load(main_dir + filename + '_index_Label' +'.npy')
            Query = np.load(main_dir + filename +'_query_Embedding' +'.npy')
            QueryID = np.load(main_dir + filename +'_query_ID' + '.npy')
            QueryLabel = np.load(main_dir + filename + '_query_Label' +'.npy')
        else:
            Index = np.load(main_dir + filename +'_{}_index_Embedding'.format(i) +'.npy')
            IndexID = np.load(main_dir + filename +'_{}_index_ID'.format(i) + '.npy')
            IndexLabel = np.load(main_dir + filename + '_{}_index_Label'.format(i) +'.npy')
            Query = np.load(main_dir + filename +'_{}_query_Embedding'.format(i) +'.npy')
            QueryID = np.load(main_dir + filename +'_{}_query_ID'.format(i) + '.npy')
            QueryLabel = np.load(main_dir + filename + '_{}_query_Label'.format(i) +'.npy')
        
    #print(collections.Counter(IndexLabel))
    #print(collections.Counter(QueryLabel))
    
        print(QueryID[:5], QueryLabel[:5])
        if normed:
                
            print('Normalizing the dataset')
            temp = time.perf_counter()
            Index= Index/np.linalg.norm(Index, axis=1).reshape(-1, 1)
            Query= Query/np.linalg.norm(Query, axis=1).reshape(-1, 1)
            normtime = time.perf_counter() - temp
            print('Done')   
            
        

        



        
        probe = 1
        indexclasses = collections.Counter(IndexLabel)
        queryclasses = collections.Counter(QueryLabel)
        allCandidates = 0
        for label in indexclasses.keys():
            allCandidates += indexclasses[label]*queryclasses[label]

        

        index, indextime, indexmemory = createIndex(Index, indexstring,metric)
        
        


        nn = len(Index)
           
        print ("Now testing NN: ", nn)
        D,I,searchtime, searchmemory = search(Query, index, nn, probe)
        
        
        fig, ax = plt.subplots(constrained_layout=True, figsize=(10,7))
        
        for query in range(5):

            n, bins, patches = ax.hist(D[query], bins = 50, histtype='step', log=True)
        ax.set_xlabel("distance distribution")
        ax.set_ylabel("counts")
        
        setting = ''
        if metric == faiss.METRIC_L2:
            setting += '_L2'
        else:
            setting += '_IP'
        if normed:
            setting += '_normed'
        if i < 10:
            plt.savefig('DistanceDistribution_'+filename+'_{}_'.format(i)+setting+'.pdf', bbox_inches='tight')
        else:
            plt.savefig('DistanceDistribution_'+filename+'_allLabels_'+setting+'.pdf', bbox_inches='tight')
    
        
        index2, tmp1, tmp2 = createIndex(Query[1:5], indexstring, metric)
        nn = 4
        
        D2,I2, tmp1, tmp2 = search(Query[0:1], index, nn, probe)
        
        
        #fig2, ax2 = plt.subplots(constrained_layout=True, figsize=(10,7))
        print(D2)

        #n, bins, patches = ax2.hist(D2[0], bins = 4, histtype='step')
        #ax.set_xlabel("distance")
        #ax.set_ylabel("counts")
        
        if i <10:
            outfile.write('Label {} \n QueryIDs: '.format(i)+ ' ,'.join(np.array(QueryID[:5],dtype=str)) + '\nDistances: ' + ' ,'.join(np.array(D2[0], dtype=str)) + '\n')
        else:
            outfile.write('All Labels \n QueryLabels: '+ ' ,'.join(np.array(QueryLabel[:5],dtype=str)) +'\n QueryIDs: '+ ' ,'.join(np.array(QueryID[:5],dtype=str)) + '\nDistances: ' + ' ,'.join(np.array(D2[0], dtype=str)) + '\n')

    outfile.close() 
    plt.show()            
            


