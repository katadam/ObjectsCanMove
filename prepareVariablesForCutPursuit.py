import open3d as o3d 
from io import StringIO
from pypcd import pypcd
import numpy as np
import faiss
import os
import glob
import sys


def computeProbabilityOfsupervoxel(init,new,label):
 
    new = np.ascontiguousarray(new)
    init = np.ascontiguousarray(init)
    d = init.shape[1]
    res = faiss.StandardGpuResources()  # use a single GPU
    index_flat = faiss.IndexFlatL2(d)  # build a flat (CPU) index
    
    # make it a flat GPU index
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(init.astype(np.float32))  
    k = 1                         # we want to see 4 nearest neighbors
 
    D, I = gpu_index_flat.search(new.astype(np.float32), k)  # actual search
    
    #test here if it needs to be less than <0.20
    
    #Inn = I[D<0.20] here to testing
    Inn = I
    In = np.unique(Inn)
    
    return label[In], I
    
    #init before want to macth before with new, no we want to match allChangedPoints (before/init) with pcdCloud new (pcdcloud)
    
    
def saveMinCut(pathR):
    pc = pypcd.PointCloud.from_path( pathR + '/supervoxel.pcd')
    adjMatrix = np.loadtxt(pathR +'/supervoxel_adj.txt')
    adjMatrix = adjMatrix.astype(int)
    supervoxels = pc.pc_data['label']

    #when supervoxels are optimized
    uniqueSuper = np.unique(supervoxels)
    y = np.zeros((2,np.shape(uniqueSuper)[0]))
    

    label = np.asarray(pc.pc_data['label'])
    x =  np.asarray(pc.pc_data['x'])
    y =  np.asarray(pc.pc_data['y'])
    z =  np.asarray(pc.pc_data['z'])
    pointsF = np.vstack((x,y,z))
    unq = np.unique(label)
    
   
    pointsF = np.transpose(pointsF)

    changedPoints = np.loadtxt(pathR +'/reprojected_10.xyz')
    [closest, I] = computeProbabilityOfsupervoxel(pointsF,changedPoints, label)
    potentialProbability = np.zeros((len(unq),2))
    for i in range(len(unq)):
        potentialProbability[i,0] = (np.count_nonzero(closest == unq[i]))/(np.count_nonzero(label == unq[i]))
    y = np.transpose(potentialProbability)
    
   
    
    

    #attribute voxels to transformations
    os.chdir( pathR + '/transform_0.10__0.5/')
    myFiles = glob.glob('*.npy')
    init = pointsF

    length = len(myFiles)
    path = pathR +  '/labelsSupervoxels16_15transforms/'
    isExist = os.path.exists(path)
    if ( isExist is False):
        os.mkdir(path)

    for i in range(1,int(length/2)):
        consistent = np.load( pathR + '/transform_0.10__0.5/consistentRescan/consistenRescan_DGCNN_2mutual' + str(i) + '.npy')
        consistent = np.ascontiguousarray(consistent)
        init = np.ascontiguousarray(init)
        d = init.shape[1]
        res = faiss.StandardGpuResources()  # use a single GPU
        index_flat = faiss.IndexFlatL2(d)  # build a flat (CPU) index

        # make it a flat GPU index
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        gpu_index_flat.add(init.astype(np.float32))  
        k = 1                         # we want to see 4 nearest neighbors

        D, I = gpu_index_flat.search(consistent.astype(np.float32), k)  # actual search
        index = D<0.10

        Ifinal = I[index]
        consistentLabels = label[Ifinal]
        np.save(pathR +  '/labelsSupervoxels16_15transforms/' + str(i) + '.npy',consistentLabels)



    #compute edges and solve
    firstRowAdj = adjMatrix[:,0]
    os.chdir(pathR +  '/labelsSupervoxels16_15transforms/')
    myFiles = glob.glob('*.npy')
    adj = np.array([0])
    transformations = len(myFiles)


    yyyy = np.transpose(y)
    yy_prob = yyyy[:,0]

    count = 0
    adj = []
    first_edge = np.zeros((len(yyyy)+1,1)) 
    #print (np.shape(uniqueSuper)[0])

    for i in range(0,np.shape(uniqueSuper)[0]):

        first_edge[i] = count
        #print (changingSupervoxels[i])
        supervoxelValue = uniqueSuper[i]
        toPropAr = np.array([100000000000])
        if (yyyy[i,0] > 0):

            #just use 5 transformations
            for m in range(0,5):
                #load all consistent point with transofmations and see if they belong to adjacen voxels

                #check here is adj supervoxel is changing under the same transform

                #if supervoxel has transformation m
                m = m + 1
                consistentSupervoxel = np.unique(np.load( pathR +  '/labelsSupervoxels16_15transforms/' + str(m) + '.npy'))

                f = np.where(firstRowAdj==supervoxelValue)

                ind = adjMatrix[f,1]

                for l in range(0, len(consistentSupervoxel)):

                    if (consistentSupervoxel[l] in ind) and (consistentSupervoxel[l] not in toPropAr):
                        finding = np.where(uniqueSuper == consistentSupervoxel[l])
                        #print (finding[0])
                        adj = np.hstack((adj,finding[0]))
                        count = count + 1
                        toPropAr = np.hstack((toPropAr,consistentSupervoxel[l]))



    first_edge[i+1] = count 

    
    
    np.save( pathR +  '/y_existing.npy',y)
    np.save(pathR +'/first_edge_existing.npy',first_edge)
    np.save(pathR +'/adj_existing.npy',adj)



file1 = open('validation-only-rescan.txt')
Lines = file1.readlines()

if __name__ == '__main__':
    for i in range(0, len(Lines)):

        pathS = sys.path[0] + '/' + Lines[i][:len(Lines[i])-1]
        print ("Computing variables for", pathS)
        saveMinCut(pathS)
