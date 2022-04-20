
#!/usr/bin/python

import numpy as np
import collections
import random

def divide (Emb, ID, Label, number, mode='last'):
    """
    INPUT
    type numpy array
        Emb: the embedding vectors
        ID: their IDs
        Label: their labels
    type int
        number: number of queries per distinct label
    type string
        mode: possibilities are
            last -> start from the end
            first -> start from the beginning
            random -> choose randomly every time
    
    OUTPUT
    Index and Query Embeddings, IDs and Labels.
    Queries are chosen in such a way, that for each label there are 'number' of vectors.
    Depending on the mode they are taken from the beginning, end or randomly from the original array. 
    
    """
    query = []
    queryID = []
    queryLabel = []
    numLabels = len(set(Label))
    counter = collections.Counter()
    tracker = []
    if mode == 'last':
        i = len(Emb)-1
        step = -1
    elif mode == 'first':
        i = 0
        step = 1
    elif mode == 'random':
        i = random.randrange(len(Emb))
        step=0
    else:
        print("The mode you have given is not defined, possible modes are 'last', 'first' and 'random'")
        return None
    while len(list(counter.elements())) < number*numLabels:
        if counter[Label[i]] < number:
            query.append(Emb[i])
            queryID.append(ID[i])
            queryLabel.append(Label[i])
            tracker.append(i)
            counter[Label[i]]+=1
        i += step
        if mode == 'random':
            newi = random.randrange(len(Emb))
            while newi in tracker:
                newi = random.randrange(len(Emb))
                print('New i had to be sampled')
            i = newi
    index = np.delete(Emb, tracker, 0)
    indexID = np.delete(ID, tracker)
    indexLabel = np.delete(Label, tracker)
    print('Query- and Indexdata have been sampled')
    return index, indexID, indexLabel, np.array(query), np.array(queryID), np.array(queryLabel)
    
def separateClass (classlabel, Emb, ID, Label):
    classEmb = []
    classID = []
    classLabel = []
    for i in range(len(ID)):
        if Label[i] == classlabel:
            classEmb.append(Emb[i])
            classID.append(ID[i])
            classLabel.append(Label[i])
    return classEmb, classID, classLabel
    

                

if __name__ == '__main__':
    main_dir = ''      #folder of the datasets and embeddings
    method = 'siamese'                       
    vectorsize = 512
    filename = method + '_{}'.format(vectorsize)
    
    indexquery = True
    classes = True

    Emb = np.load(main_dir + filename +'_Embedding' +'.npy')
    ID = np.load(main_dir + filename +'_ID' + '.npy')
    Label = np.load(main_dir + filename + '_Label' +'.npy')
    
    
    if classes:
        print('now separating classes')
        print('Classes: ', set(Label))
        labelNames = set(Label)
        for i in set(Label):
            classEmb, classID, classLabel = separateClass(i, Emb, ID, Label)
            
            Index, IndexID, IndexLabel, Query, QueryID, QueryLabel = divide(classEmb, classID, classLabel, 5, 'first')
        
            np.save( main_dir + filename +'_{}_index_ID'.format(i), IndexID)
            np.save( main_dir + filename +'_{}_index_Label'.format(i), IndexLabel)
            np.save( main_dir + filename +'_{}_index_Embedding'.format(i), Index)
            np.save( main_dir + filename +'_{}_query_ID'.format(i), QueryID)
            np.save( main_dir + filename +'_{}_query_Label'.format(i), QueryLabel)
            np.save( main_dir + filename +'_{}_query_Embedding'.format(i), Query)
            
    
    if indexquery:
        Index, IndexID, IndexLabel, Query, QueryID, QueryLabel = divide(Emb, ID, Label, 500, 'first')
        
        np.save( main_dir + filename +'_index_ID', IndexID)
        np.save( main_dir + filename +'_index_Label', IndexLabel)
        np.save( main_dir + filename +'_index_Embedding', Index)
        np.save( main_dir + filename +'_query_ID', QueryID)
        np.save( main_dir + filename +'_query_Label', QueryLabel)
        np.save( main_dir + filename +'_query_Embedding', Query)
    
    
    
    
