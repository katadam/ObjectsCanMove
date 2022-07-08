import os
import open3d as o3d
import numpy as np
import sys



#for supervoxels do the same but firslty calculate the grid of
def metrics(pathS):

    meanAll = np.empty((0))
    name = pathS + '/labels.instances.colored.normals.pcd' 
    pcd = o3d.io.read_point_cloud(name)
    grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,0.10)
 
    # load 
    points = np.loadtxt( pathS  +'/final_changing_points.xyz')
    allVoxels = np.empty((0,3))

 
    
    for i in range(len(points)):
        voxel = grid.get_voxel(points[i,:3])
        allVoxels = np.vstack((voxel,allVoxels))
    
    dt = np.dtype((np.void, allVoxels.dtype.itemsize *  allVoxels.shape[1]))
    b = np.ascontiguousarray( allVoxels).view(dt)
    unq, cnt = np.unique(b, return_counts=True)
    unq = unq.view( allVoxels.dtype).reshape(-1,  allVoxels.shape[1])
    
    finalSolut  = unq
    count = 0
    
    
    
    
    points = np.loadtxt(pathS + '/groundTruthChangedNEW_withoutaddedobjects.xyz')
    allVoxels = np.empty((0,3))

    
    for i in range(0,len(points)):
        voxel = grid.get_voxel(points[i,:])
        allVoxels = np.vstack((voxel,allVoxels))
    
    dt = np.dtype((np.void, allVoxels.dtype.itemsize *  allVoxels.shape[1]))
    b = np.ascontiguousarray( allVoxels).view(dt)
    unq, cnt = np.unique(b, return_counts=True)
    unq = unq.view( allVoxels.dtype).reshape(-1,  allVoxels.shape[1])
    
    gtSolut  = unq
    count = 0
    
    
    
    
    for i in range(len(finalSolut)):
        voxel = finalSolut[i,:]
        if any((gtSolut[:]== voxel).all(1)):
            count = count + 1
    if len(gtSolut)>0:
        recall = count/len(gtSolut)
    else:
        recall = 1
        
    return recall
           

if __name__ == '__main__':
    file1 = open('validation-only-rescan.txt')
    Lines = file1.readlines()
    allObjectsDi = 0
    allObjectsG = 0
    allPercentPerScan = np.zeros((110))
    allRecall = np.zeros((110))
    for i in range(0, len(Lines)):

        
        pathS =  sys.path[0]+ '/' + Lines[i][:len(Lines[i])-1] 
        print (pathS)
        recall = metrics(pathS)
        allRecall[i] = recall
        print (recall) 

    print ("Mean Recall is ",np.mean(allRecall))        





