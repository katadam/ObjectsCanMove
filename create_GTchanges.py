import json
import numpy as np
from plyfile import PlyData,PlyElement
import sys


def retrievePoints(whereObjectIdnex,plydata):
    
    resultsOfObject = np.asarray(whereObjectIdnex)
    
    
    finalCoords = np.zeros((3,np.shape(resultsOfObject)[1])) 
    finalCoords = np.zeros((3,np.shape(resultsOfObject)[1])) 
    finalCoords = np.zeros((3,np.shape(resultsOfObject)[1])) 
    
    finalCoords[0][:] = plydata.elements[0].data['x'][resultsOfObject]
    finalCoords[1][:] = plydata.elements[0].data['y'][resultsOfObject] 
    finalCoords[2][:] = plydata.elements[0].data['z'][resultsOfObject]
    return finalCoords


with open('3RScan.json') as json_file:
    data = json.load(json_file)
    i = 0
    for item in data:
        if (item['type'] == 'validation'):
            for data_item in item['scans']:
                    removed = np.asarray(data_item['removed'])
                   
                    ref =  item['reference']
                    dd = data_item['reference']
                    print ("Now Processing ",dd)

                    
                    fR = open(sys.path[0] + '/' + dd + '/semseg.json')
                    fS = open(sys.path[0] + '/' + ref + '/semseg.json')

                    dataS = json.load(fS)
                    dataR = json.load(fR)
                    plydataS = PlyData.read(sys.path[0] + '/' + ref + '/labels.instances.annotated.ply')
                    plydataR = PlyData.read(sys.path[0] + '/' +  dd + '/labels.instances.annotated.ply')
                    
                    plydataRC = PlyData.read(sys.path[0] + '/' + dd + '/labels.instances.annotated.align.ply')
                    scan = plydataS.elements[0].data['objectId']
                    scan = scan.astype(np.int)
                    rescan = plydataR.elements[0].data['objectId']
                    rescan = rescan.astype(np.int)
                    finalCoordsSF = np.empty((3,0))
                    finalCoordsSR = np.empty((3,0))
                    
                    
                    scan = plydataS.elements[0].data['objectId']
                    scan = scan.astype(np.int)
                    rescan = plydataR.elements[0].data['objectId']
                    rescan = rescan.astype(np.int)
                   
                    labelsS = np.empty((1,1),dtype=np.int)
                    labelsR = np.empty((1,1),dtype = np.int)
                    for i in dataS['segGroups']: 
                        labelsS = np.vstack((labelsS,i['id']))



                    for i in dataR['segGroups']: 
                        labelsR = np.vstack((labelsR,i['id']))   

                    
                    
                    for changed in data_item['rigid']:
                       
                        if changed['instance_reference']:

                            in_ref = np.asarray(changed['instance_reference'])
                            in_rescan = np.asarray(changed['instance_rescan'])
   

                            whereObjectIdnexS = np.where(scan == in_ref)
                            whereObjectIdnexR = np.where(rescan == in_rescan)

                            resultsOfObjectS = np.asarray(whereObjectIdnexS)
                            resultsOfObjectR = np.asarray(whereObjectIdnexR)

                            finalCoordsS = np.zeros((3,np.shape(resultsOfObjectS)[1])) 
                            finalCoordsR = np.zeros((3,np.shape(resultsOfObjectR)[1])) 

                            finalCoordsS[0][:] = plydataS.elements[0].data['x'][resultsOfObjectS]
                            finalCoordsS[1][:] = plydataS.elements[0].data['y'][resultsOfObjectS] 
                            finalCoordsS[2][:] = plydataS.elements[0].data['z'][resultsOfObjectS]



                            finalCoordsR[0][:] = plydataRC.elements[0].data['x'][resultsOfObjectR]
                            finalCoordsR[1][:] = plydataRC.elements[0].data['y'][resultsOfObjectR] 
                            finalCoordsR[2][:] = plydataRC.elements[0].data['z'][resultsOfObjectR]

                        
                            finalCoordsSF = np.hstack((finalCoordsSF,finalCoordsS))
                            finalCoordsSR = np.hstack((finalCoordsSR,finalCoordsR))
                        
                    
                        if data_item['nonrigid']:
                            for changed in data_item['nonrigid']:
                              
                                if changed not in removed:
                                    in_ref = np.asarray(changed)
                                    in_rescan = np.asarray(changed)

                                    whereObjectIdnexS = np.where(scan == in_ref)
                                    whereObjectIdnexR = np.where(rescan == in_rescan)

                                    resultsOfObjectS = np.asarray(whereObjectIdnexS)
                                    resultsOfObjectR = np.asarray(whereObjectIdnexR)

                                    finalCoordsS = np.zeros((3,np.shape(resultsOfObjectS)[1])) 
                                    finalCoordsR = np.zeros((3,np.shape(resultsOfObjectR)[1])) 

                                    finalCoordsS[0][:] = plydataS.elements[0].data['x'][resultsOfObjectS]
                                    finalCoordsS[1][:] = plydataS.elements[0].data['y'][resultsOfObjectS] 
                                    finalCoordsS[2][:] = plydataS.elements[0].data['z'][resultsOfObjectS]

                                    finalCoordsR[0][:] = plydataRC.elements[0].data['x'][resultsOfObjectR]
                                    finalCoordsR[1][:] = plydataRC.elements[0].data['y'][resultsOfObjectR] 
                                    finalCoordsR[2][:] = plydataRC.elements[0].data['z'][resultsOfObjectR]


                                    finalCoordsSF = np.hstack((finalCoordsSF,finalCoordsS))
                                    finalCoordsSR = np.hstack((finalCoordsSR,finalCoordsR))
                            

                   #to check which objects have been added uncomment these lines
                        
                    #for i in range(len(labelsS)):
                     #   if labelsS[i] not in labelsR and labelsS[i] not in removed:
                      #      #print (labelsS[i])
                      #      whereObjectIdnexS = np.where(scan == labelsS[i])
                      #      finalCoordsS = retrievePoints(whereObjectIdnexS,plydataS)
                      #      finalCoordsSF = np.hstack((finalCoordsSF,finalCoordsS)) 


                    #for i in range(len(labelsR)):
                     #   if labelsR[i] not in labelsS and labelsR[i] not in removed:
                         #   whereObjectIdnexR = np.where(rescan == labelsR[i])
                         #   finalCoordsR = retrievePoints(whereObjectIdnexR,plydataRC)
                         #   finalCoordsSR = np.hstack((finalCoordsSR,finalCoordsR)) 
        
                 
                    finalNonStaticPointsS = finalCoordsSF.transpose()
                    finalNonStaticPointsR = finalCoordsSR.transpose()
                    np.savetxt(sys.path[0] + '/' + dd + '/groundTruthChanged_withoutaddedobjects.xyz',finalNonStaticPointsR )    
             

