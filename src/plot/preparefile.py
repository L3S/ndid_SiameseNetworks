
#!/usr/bin/python

import sys
import struct
import numpy as np
import pandas as pd
import time
import collections


def fileread(method, filename, vectorsize, deliminator = ';', ToDelete = set([])):
    """
    reads in a file and returns data from it as a list.
    
    MANDATORY INPUT:
        method (string) -> sift, hsv or siamese
        filename (string)
        vectorsize -> size of embedding vectors
        
    OPTIONAL INPUT:
        deliminator (string, default ';') -> what character separates the
            columns in the file
        ToDelete (set, default empty set) -> IDs of vectors, which should not be included in further
            analysis

    OUTPUT:
        3 np.arrays: ID, Label, Embedding (third column of the input file, name is chosen depending on the method)
        nonID (list) IDs of those vectors which were not included in the output arrays

    """
    print('reading now in file '+filename)   
    file = pd.read_csv(filename, sep = deliminator, engine = 'python')
    print('now read in '+ filename)
    if method == 'sift':
        rawdata = file['SIFT descriptors']
    if method == 'hsv':
        rawdata = file['HSV vector']
    if method == 'siamese':
        rawdata = file['Siamese Embeddings']
    data = []
    IDs = file['ID'].to_numpy()
    Labels = file['Label'].to_numpy()
    longerVectors=collections.Counter()
    shorterVectors = collections.Counter()
    noneVectors= collections.Counter()
    length = collections.Counter()
    noneID = []
    for i in range(len(rawdata)):  
        vector = rawdata[i]
        if not vector == '' and not pd.isnull(vector) and not vector == 'None' and not i in ToDelete:
            vector = vector.split(',')
            length[len(vector)] +=1
            if len(vector)<vectorsize:
                shorterVectors[Labels[i]] +=1
                noneID.append(i)
            elif len(vector)>vectorsize:
                longerVectors[Labels[i]] +=1
                noneID.append(i)
            #while len(vector)<vectorsize:
                #vector.append('0')
            else:
                data.append(vector[:vectorsize])
        else:
            noneID.append(i) 
            noneVectors[Labels[i]]+=1
    IDs = np.delete(IDs, noneID)
    Labels = np.delete(Labels, noneID)
    print(length)
    
    print(longerVectors, shorterVectors, noneVectors)
    print(collections.Counter(Labels))
    return IDs, Labels, np.array(data, dtype='float32'), noneID
        
        
#Main infos for starting
main_dir = '../Data/' #folder of the datasets and embeddings
method = 'siamese'                       
vectorsize = 512
deli = ';'                           #delimiter in the files


starttime = time.perf_counter()
#creating the filenames from the infos given above
mainfile =  method + '_{}'.format(vectorsize)

deleted = []
deleted = np.load('sift_{}'.format(vectorsize) + '_deleted' +'.npy')
    
#reading in the files
deleted = set(deleted)
IDs, Labels, Embeddings, noneID = fileread(method, main_dir+mainfile+'.csv', vectorsize, ToDelete=deleted)
print(len(noneID))
print(Labels)


np.save( mainfile+'_ID', IDs)
np.save( mainfile+'_Label', Labels)
np.save( mainfile+'_Embedding', Embeddings)
#np.save( mainfile+'_deleted', np.array(noneID))
stoptime = time.perf_counter()
print('Reading and saving the file took {}s'.format(stoptime-starttime))
