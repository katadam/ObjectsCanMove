import os
import open3d as o3d
import numpy as np
import faiss
import glob
import numpy as np
import os
import glob
import open3d as o3d
from scipy.ndimage.measurements import label
import open3d as o3d
import cc3d
import numpy as np
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.ckdtree import cKDTree
from mpl_toolkits.mplot3d import Axes3D
import os
import sys



def computeAcc(componPoints_noGT , componPoints):

    d = componPoints.shape[1]
    res = faiss.StandardGpuResources() 
    index_flat = faiss.IndexFlatL2(d)  # use a single GPU
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
     # build a flat (CPU) index
    gpu_index_flat.add(componPoints.astype(np.float32))         # add vectors to the index

    k = 1
    D, I = gpu_index_flat.search(componPoints_noGT.astype(np.float32), k)  # actual search

    count = np.where(D<0.0001)
 
    accuracy =  np.shape(count)[1] / np.shape(componPoints_noGT)[0]

    complet =  np.shape(count)[1] / np.shape(componPoints)[0]

    return accuracy, complet





def computeMetrics(pathR, acc, comp, countObj):
    localCount = 0
    os.chdir( pathS + '/gtCom/')
    fGt = len(glob.glob("*.npy"))
    gtObjectsDetect = np.zeros((fGt,3))
   
    files = glob.glob("*.npy")
 
    for j in range(1,fGt):
     
        countF = 0
        maxAccu = 0
        maxComplet = 0
        maxMesosOros = 0
        componPoints = np.loadtxt(pathS +'/gtCom/'+ str(j) + '.xyz')

        localCount = 0
        if np.shape(componPoints)[0] > 1000:
            localCount = localCount + 1
            countObj = countObj + 1
            for filename in os.listdir(+pathS +'/computedConnected/'):
        
                if filename!='0comp.xyz':
                    
                    componPoints_noGT = np.loadtxt(pathS +'/computedConnected/'+filename)
                    if np.shape(componPoints_noGT)[0]>500:
                        accuracy, completeness = computeAcc(componPoints_noGT , componPoints)
                        if completeness > 1:
                            completeness = 1
                        if accuracy>1:
                            accuracy = 1
                        if (completeness  > maxComplet):
                            maxMesosOros = accuracy+completeness 
                            maxAccu = accuracy
                            maxComplet = completeness
                else:
                    countF = countF + 1

            acc = np.vstack((acc,maxAccu))
            comp = np.vstack((comp,maxComplet))
    return acc, comp, countObj


if __name__ == '__main__':
    file1 = open('validation-only-rescan.txt')
    Lines = file1.readlines()
    allObjectsDi = 0
    allObjectsG = 0
    allPercentPerScan = np.zeros((110))
    acc = np.array([1])
    comp = np.array([1])
    countObj=0
    for i in range(0, len(Lines)):

        
        pathS =  sys.path[0]+ '/' + Lines[i][:len(Lines[i])-1] 
        print ("Metrics for ", pathS," are computed" )
        acc, comp, countObj = computeMetrics(pathS, acc, comp, countObj)
