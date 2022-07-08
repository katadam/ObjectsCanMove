import numpy as np
import faiss
import sys
import json


def computeMatcheswithNerarNeihbor(pathS,pathR):
 
    
    pointsRescan = np.load(pathR +  '/points.npy')
    descR = np.load(pathR +   '/features.npy')
    descS = np.load(pathS +  '/features.npy')
    pointsScan = np.load(pathS +  '/points.npy')
    
    

    
    if (np.shape(descS)[0]> np.shape(pointsScan)[0]):
        descS = descS[1:,:]
    if (np.shape(descR)[0]> np.shape(pointsRescan)[0]):
        descR = descR[1:,:]
        
 
        

    scan = np.ascontiguousarray(descS, dtype=np.float32)
    rescan = np.ascontiguousarray(descR, dtype=np.float32)

    d = rescan.shape[1]
    res = faiss.StandardGpuResources()  # use a single GPU
    index_flat = faiss.IndexFlatL2(d)  # build a flat (CPU) index
    # make it a flat GPU index
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(scan)         # add vectors to the index
    k = 1                         # we want to see 4 nearest neighbors
    D, I = gpu_index_flat.search(rescan, k)  # actual search


    corr1 = np.zeros((rescan.shape[0],2), int)
    corr1[:,0] = np.arange(rescan.shape[0])
    corr1[:,1] = I[:,0]



    d = rescan.shape[1]
    res = faiss.StandardGpuResources()  # use a single GPU

    index_flat = faiss.IndexFlatL2(d)  # build a flat (CPU) index

    # make it a flat GPU index
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

    gpu_index_flat.add(rescan)         # add vectors to the index
    k = 2                         # we want to see 4 nearest neighbors
    D1, I22 = gpu_index_flat.search(scan, k)  # actual search
    I1 = I22[:,0]
    I11 = I22[:,1]


    corr2 = np.zeros((scan.shape[0],2), int)
    corr2[:,0] = np.arange(scan.shape[0])
    corr2[:,1] = I1
    
    corr22 = np.zeros((scan.shape[0],2), int)
    corr22[:,0] = np.arange(scan.shape[0])
    corr22[:,1] = I11
       

    emptylist = np.zeros((1,2),np.int)

    
    
    for i in range(corr1.shape[0]):

        if (corr1[i,1] == corr2[corr1[i,1],0]) or (corr1[i,1] == corr22[corr1[i,1],0]):

            arr = np.array([[corr1[i,0],corr1[i,1]]])
            emptylist = np.append(emptylist,arr,axis=0)


    pointsA = [0,0,0]
    pointsK = [0,0,0]

    for i in range(0,emptylist.shape[0]):
        pointsA = np.vstack((pointsA,pointsRescan[emptylist[i][0]]))
        pointsK = np.vstack((pointsK,pointsScan[emptylist[i][1]]))

    
    np.savetxt( pathR + '/matched_scan', pointsK)
    np.savetxt(pathR + '/matched_rescan', pointsA)
    distances = pointsK - pointsA
    distancesN = np.linalg.norm(distances, axis = 1)




with open('3RScan.json') as json_file:
    data = json.load(json_file)
    for item in data:
        if (item['type'] == 'validation'):
            for data_item in item['scans']:
                    removed = np.asarray(data_item['removed'])
                    ref =  item['reference']
                    dd = data_item['reference']
                    pathS = sys.path[0]+ '/' +  ref
                    pathR = sys.path[0]+ '/' +  dd
                    print ("computing matches between ", pathS, " and ", pathR)
                 
                    computeMatcheswithNerarNeihbor(pathS,pathR)
