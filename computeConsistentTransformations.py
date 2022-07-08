from numpy import linalg
import os
import open3d as o3d
import open3d as open3d
import numpy as np
import trimesh
from plyfile import PlyElement
from plyfile import PlyData, PlyElement
from random import randrange
import json
import sys

def createTransformations(pathR):   
    pa = np.loadtxt(pathR + '/matched_rescan')
    pp = np.loadtxt(pathR +'/matched_scan')



    
    os.mkdir(pathR + '/transform_0.10__0.5')
    os.mkdir(pathR + '/transform_0.10__0.5/consistentRescan')
    
    distances = pa - pp
    distancesN = np.linalg.norm(distances, axis = 1)

    
    #check here the ransac threshold to eliminate points as static
    index = distancesN>0.20

    finalDist = distancesN[index]

    pp_scan = pp[index,:3]
    pa_rescan = pa[index,:3]
    matches = np.zeros((pp_scan.shape[0],2),np.int)
    matches[:,0] = np.arange(pp_scan.shape[0])
    matches[:,1] = np.arange(pp_scan.shape[0])

    #source = rescan compute from rescan to scan
    corres = o3d.utility.Vector2iVector(matches)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(pa_rescan)

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(pp_scan)

    count10 = 0

    while (pa_rescan.shape[0] >= 3):
   
        result = open3d.pipelines.registration.registration_ransac_based_on_correspondence(source, target, corres, max_correspondence_distance=0.5)

        count10 = count10 +1 
        np.save(pathR + '/transform_0.10__0.5/transform_DGCNN_2mutual'+ str(count10), result.transformation)

        f = np.asarray(result.correspondence_set)
        f1 = f[:,0]
        f2 = f[:,1]
        pointsA = pa_rescan[f1,:]
        pointsB = pp_scan[f2,:]
        
        np.save( pathR + '/transform_0.10__0.5/consistentRescan/consistenRescan_DGCNN_2mutual'+ str(count10), pointsA)
        np.save( pathR + '/transform_0.10__0.5/consistenScan_DGCNN_2mutual'+ str(count10), pointsB)
     

        pp_scan = np.delete(pp_scan, f2, axis = 0)
        pa_rescan = np.delete(pa_rescan, f1, axis = 0)
  


        target.points = o3d.utility.Vector3dVector(pp_scan)
        source.points = o3d.utility.Vector3dVector(pa_rescan)

        corres = o3d.utility.Vector2iVector(matches)
        matches = np.zeros((pp_scan.shape[0],2),np.int)
        matches[:,0] = np.arange(pa_rescan.shape[0])
        matches[:,1] = np.arange(pp_scan.shape[0])
        corres = o3d.utility.Vector2iVector(matches)


if __name__ == '__main__':
    with open('3RScan.json') as json_file:
        data = json.load(json_file)
        for item in data:
            if (item['type'] == 'validation'):
                for data_item in item['scans']:
                    removed = np.asarray(data_item['removed'])
                    ref =  item['reference']
                    dd = data_item['reference']
                    pathR = sys.path[0] + '/' + dd
                    print ("computing transformations for", pathR)
                    createTransformations(pathR)
                    
