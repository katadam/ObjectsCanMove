#------------------------------------------------------------------------#
  #  script for illustrating cp_pfdr_d1_lsx on labeling of 3D point cloud  #
  #------------------------------------------------------------------------#
# Reference: H. Raguet and L. Landrieu, Cut-Pursuit Algorithm for Regularizing
# Nonsmooth Functionals with Graph Total Variation, International Conference on
# Machine Learning, PMLR, 2018, 80, 4244-4253
#
# Camille Baudoin 2019
import sys
import os 
import numpy as np
import scipy.io
import time
from io import StringIO
from pypcd import pypcd
import sys

sys.path.append("/cut-pursuit/parallel-cut-pursuit/python/wrappers")

from cp_pfdr_d1_lsx import cp_pfdr_d1_lsx 



def solveminCutPutsuit_final(pathR):
    pfdr_rho = 1
    homo_d1_weight = 0.0009
    y = np.load( pathR +  '/y_existing.npy')
    first_edge = np.load(pathR +'/first_edge_existing.npy')
    adj = np.load(pathR +'/adj_existing.npy')

    first_edge = np.squeeze(first_edge)
    first_edge = np.uint32(first_edge)
    adj = np.uint32(adj)
    y1 = y[0,:]
    indexChanged = np.where(y1>0)
    indexNonChanged = np.where(y1==0)

    y[0,indexChanged] = 0.9
    y[0,indexNonChanged] = 0.5
    y[1,indexNonChanged] = 0.5
    loss = 0.1


    ###  solve the optimization problem  ###
    it1 = time.time()
    Comp, rX, it = cp_pfdr_d1_lsx(loss, y, first_edge, adj,edge_weights=homo_d1_weight)
    it2 = time.time()
    x = rX[:,Comp] # rX is components values, Comp is components assignment


    pc = pypcd.PointCloud.from_path( pathR + '/supervoxel.pcd')
    alllabels = np.asarray(pc.pc_data['label'])
    label = np.unique(np.asarray(pc.pc_data['label']))
    changed = x[0,:]
    indexChanged = np.where(changed > 0.5)
    pointsFinal = []
    supervoxelLabelChanged = label[indexChanged]
    pointsFinal = np.zeros((1,3))


    xx =  np.asarray(pc.pc_data['x'])
    yy =  np.asarray(pc.pc_data['y'])
    zz =  np.asarray(pc.pc_data['z'])
    pointsF = np.vstack((xx,yy,zz))
    pointsA = np.transpose(pointsF)

    for i in range(0, len(supervoxelLabelChanged)):
        indexPoints = np.where(alllabels == supervoxelLabelChanged[i])
        pointsToAdd = np.squeeze(pointsA[indexPoints,:])
        pointsFinal = np.vstack((pointsFinal, pointsToAdd))

    pointsFinal = pointsFinal[1:,:]


    np.savetxt(pathR  +'/final_changing_points.xyz', pointsFinal)
    np.save( pathR + '/final_changing_points.npy', pointsFinal)

    pointsFinal1 = np.zeros((1,3))
    adjAll = np.unique(adj)


file1 = open('validation-only-rescan.txt')
Lines = file1.readlines()

if __name__ == '__main__':
    for i in range(0, len(Lines)):

        pathS = sys.path[0] + '/'+ Lines[i][:len(Lines[i])-1]
        print ("Solving graph for", pathS)

        solveminCutPutsuit_final(pathS)
