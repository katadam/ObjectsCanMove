import open3d as o3d
from array import array
import numpy as np
import sys
import json

def returnPCD(name, pathToStore):
    mesh = o3d.io.read_triangle_mesh(name)
    vertices = mesh.vertices
    kk = o3d.geometry.PointCloud()
    kk.points = vertices
    
    kk.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    kk.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
    nameToStore = pathToStore + '/labels.instances.colored.normals.pcd' 
    print (nameToStore, "created")
    o3d.io.write_point_cloud(nameToStore, kk)
    

if __name__ == '__main__':
    with open('3RScan.json') as json_file:
        data = json.load(json_file)
        for item in data:
            if (item['type'] == 'validation'):
                for data_item in item['scans']:
                    removed = np.asarray(data_item['removed'])
                    ref =  item['reference']
                    dd = data_item['reference']
                    pathS =  sys.path[0] + '/' + ref
                    pathR = sys.path[0] + '/' + dd
                    nameR = pathS + '/mesh.refined.obj'
                    nameRescan = pathR + '/mesh.refined.align.trimesh.obj'
                    returnPCD(nameR,pathS)
                    returnPCD(nameRescan,pathR)


   

