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
import glob, os

def computeIoU(finalAr,arrayGt):
    count = 0
    
    dt = np.dtype((np.void, finalAr.dtype.itemsize *  finalAr.shape[1]))
    b = np.ascontiguousarray(finalAr).view(dt)
    unq, cnt = np.unique(b, return_counts=True)
    unq1 = unq.view( finalAr.dtype).reshape(-1, finalAr.shape[1])
    
    
    
    for i in range(0,len(finalAr)):
        box = finalAr[i,:]
        if any((arrayGt[:]== box).all(1)):
            count = count + 1

    allVoxels = np.vstack((finalAr,arrayGt))
    dt = np.dtype((np.void, allVoxels.dtype.itemsize *  allVoxels.shape[1]))
    b = np.ascontiguousarray( allVoxels).view(dt)
    unq, cnt = np.unique(b, return_counts=True)
    unq = unq.view( allVoxels.dtype).reshape(-1,  allVoxels.shape[1])
    union = len(unq)
    
 
    IoU = count/union
    return IoU

def calculateConnectedComponents(points, grid):
    
    allVoxels = np.empty((0,3))
    for i in range(len(points)):
        voxel = grid.get_voxel(points[i,:3])
        if ( voxel[2]>1):
            allVoxels = np.vstack((voxel,allVoxels))
    
    dt = np.dtype((np.void, allVoxels.dtype.itemsize *  allVoxels.shape[1]))
    b = np.ascontiguousarray( allVoxels).view(dt)
    unq, cnt = np.unique(b, return_counts=True)
    unq = unq.view( allVoxels.dtype).reshape(-1,  allVoxels.shape[1])
    
    voxels  = unq    

    maxVoxel = grid.get_max_bound()
    minVoxel = grid.get_min_bound()
    voxels = voxels.astype(np.int32)

    occ1 = int((maxVoxel[0] - minVoxel[0])/grid.voxel_size)
    occ2 = int((maxVoxel[1] - minVoxel[1])/grid.voxel_size)
    occ3 = int((maxVoxel[2] - minVoxel[2])/grid.voxel_size)
    
     
    finalSolut = np.empty((0,3))
    f = np.array(grid.get_voxels())
    voxel_S = np.zeros((len(f),3), np.int)



    for i in range (len(f)):
        voxel_S[i,:] = np.asarray(f[i].grid_index)
        
    for i in range(len(voxels)):
        voxelChecked = voxels[i,:]
        if any((voxel_S[:]== voxelChecked).all(1)):
        
            finalSolut = np.vstack((finalSolut, voxelChecked))
    

    gridToSolve = np.zeros((occ1,occ2,occ3),dtype = np.int)

    
    structure = np.array([[[0 ,0, 0],
    [0, 1 ,0],
    [0 ,0 ,0]],
    [[0, 1, 0],
    [1, 1, 1],
    [0 ,1 ,0]],
    [[0 ,0 ,0],
    [0, 1 ,0],
    [0 ,0, 0]]])

    
    for i in range(0,len(finalSolut)):

        w1 = int(finalSolut[i][0])
        w2 = int(finalSolut[i][1])
        w3 = int(finalSolut[i][2])
        if w1<occ1 and w2<occ2 and w3<occ3:
            gridToSolve[w1][w2][w3]=1


    labeled, ncomponents = label(gridToSolve, structure)
    print ("N " ,ncomponents)
    
    return labeled, ncomponents
    
    

def metrics(pathS):
    os.mkdir(pathS +'/gtCom/')
    os.mkdir(pathS +'/computedConnected/')

    meanAll = np.empty((0))
    name = pathS + '/labels.instances.colored.normals.pcd' 
    
    pcd = o3d.io.read_point_cloud(name)

    grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,0.10)
    
    pointsCh = np.loadtxt(pathS + '/groundTruthChanged_withoutaddedobjects.xyz')
    points = np.loadtxt( pathS +'/final_changing_points.xyz')
    

 
    labeled, ncomponents = calculateConnectedComponents(points, grid)
    labeledGt, ncomponentsGt = calculateConnectedComponents(pointsCh, grid)
    countforMean = 0
    count7 = 0
    countObjects = 0
    
    fGt = ncomponentsGt 
    gtObjectsDetect = np.zeros((fGt,3))


    for j in range(1,fGt):
        
        allIoUs = np.zeros((fGt-1,2))
        maxIou = 0

        [array1,array2,array3] = np.where(labeledGt == j)
        arrayGt = np.vstack((array1,array2,array3))
        arrayGt = np.transpose(arrayGt)

        componPoints = np.array([0,0,0])
        for k in range(0, np.shape(pointsCh)[0]):
            pp = pointsCh[k,:]
            volx = grid.get_voxel(pp)
            if any((arrayGt[:]== volx).all(1)):
                componPoints = np.vstack((componPoints,pp))

        
        if np.shape(componPoints)[0] > 1000:
            np.savetxt(pathS +'/gtCom/'+ str(j) + '.xyz', componPoints)
            for i in range(0,ncomponents+1):

                [array1,array2,array3] = np.where(labeled == i)

                finalAr = np.vstack((array1,array2,array3))
                finalAr = np.transpose(finalAr)
                

                ourP = np.array([0,0,0])
                for k in range(0, np.shape(points)[0]):
                    pp = points[k,:]
                    volx = grid.get_voxel(pp)
                if any((finalAr[:]== volx).all(1)):
                    ourP = np.vstack((ourP,pp))
                np.savetxt(pathS +'/computedConnected/'+ str(j) + '.xyz', ourP)
                
                interOverUn = computeIoU(finalAr,arrayGt)
                if interOverUn>maxIou:
                    maxIou = interOverUn


            if (maxIou>0.20):
                countObjects = countObjects +1

            countforMean = countforMean + maxIou
            count7 = count7 + 1
    
            #print (countObjects,count7)
    
    if count7>0:
        val = countObjects/count7
    else:
        val = 1.0
    print (val)
    
    meanAll = np.hstack((meanAll, val))
    return countObjects,count7

if __name__ == '__main__':
    file1 = open('validation-only-rescan.txt')
    Lines = file1.readlines()
    allObjectsDi = 0
    allObjectsG = 0
    allPercentPerScan = np.zeros((110))

    for i in range(0, len(Lines)):

        
        pathS =  sys.path[0]+ '/' + Lines[i][:len(Lines[i])-1] 
        print (pathS)
        countObjects,count7 = metrics(pathS)
        if count7>0:
            val = countObjects/count7
        else:
            val = 1.0


        allObjectsDi = allObjectsDi + countObjects
        allPercentPerScan[i] = val
        allObjectsG = allObjectsG + count7

    print ("Mean IoU is ",np.mean(allPercentPerScan ))  
