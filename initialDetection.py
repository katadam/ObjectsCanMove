# Load an image
from PIL import Image
import io
import numpy as np
import glob
import os
import numpy as np
import trimesh
import pyrender
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import json
import sys

csum = lambda z: np.cumsum(z)[:-1]
dsum = lambda z: np.cumsum(z[::-1])[-2::-1]
argmax = lambda x, f: np.mean(x[:-1][f == np.max(f)])  # Use the mean for ties.
clip = lambda z: np.maximum(1e-30, z)

def preliminaries(n, x):
  """Some math that is shared across multiple algorithms."""
  assert np.all(n >= 0)
  x = np.arange(len(n), dtype=n.dtype) if x is None else x
  assert np.all(x[1:] >= x[:-1])
  w0 = clip(csum(n))
  w1 = clip(dsum(n))
  p0 = w0 / (w0 + w1)
  p1 = w1 / (w0 + w1)
  mu0 = csum(n * x) / w0
  mu1 = dsum(n * x) / w1
  d0 = csum(n * x**2) - w0 * mu0**2
  d1 = dsum(n * x**2) - w1 * mu1**2
  return x, w0, w1, p0, p1, mu0, mu1, d0, d1

def GHT(n, x=None, nu=0, tau=0, kappa=0, omega=0.5, prelim=None):
  assert nu >= 0
  assert tau >= 0
  assert kappa >= 0
  assert omega >= 0 and omega <= 1
  x, w0, w1, p0, p1, _, _, d0, d1 = prelim or preliminaries(n, x)
  v0 = clip((p0 * nu * tau**2 + d0) / (p0 * nu + w0))
  v1 = clip((p1 * nu * tau**2 + d1) / (p1 * nu + w1))
  f0 = -d0 / v0 - w0 * np.log(v0) + 2 * (w0 + kappa *      omega)  * np.log(w0)
  f1 = -d1 / v1 - w1 * np.log(v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)
  return argmax(x, f0 + f1), f0 + f1

def im2hist(im, zero_extents=False):
  # Convert an image to grayscale, bin it, and optionally zero out the first and last bins.
  max_val = np.iinfo(im.dtype).max
  x = np.arange(max_val+1)
  e = np.arange(-0.5, max_val+1.5)
  assert len(im.shape) in [2, 3]
  im_bw = np.amax(im[...,:3], -1) if len(im.shape) == 3 else im
  n = np.histogram(im_bw, e)[0]
  if zero_extents:
    n[0] = 0
    n[-1] = 0
  return n, x, im_bw


def render(i):
  global _nu, _tau, _kappa, _omega
  #t, score = GHT(n, x, 1e30, 0.20, 1e-30, 0.50, prelim)
  #t, score = GHT(n, x, _nu, _tau, _kappa, _omega, prelim)
  t, score = GHT(n, x, 1.00e+9,1.58e+7,1.58e+8, 0.71, prelim)


  plt.figure(0, figsize=(16,5))
  plt.subplot(1,3,1)
  plt.imshow(im, cmap='gray')
  plt.axis('off')

  plt.subplot(1,3,2)
  plt.imshow(im_bw > t, cmap='gray', vmin=0, vmax=1)
  plt.gca().set_xticks([])
  plt.gca().set_yticks([])

  plt.subplot(1,3,3)
  normalize = lambda x : (x - np.min(score)) * np.max(n) / (np.max(score) - np.min(score))
  plt.plot((x[:-1] + x[1:])/2, normalize(score))
  plt.scatter(t, normalize(score[int(t)]))
  plt.bar(x, n, width=1)
  plt.gca().set_yticks([]);
  
 
def update(nu=None, tau=None, kappa=None, omega=None):
  global _nu, _tau, _kappa, _omega
  _nu = nu or _nu
  _tau = tau or _tau
  _kappa = kappa or _kappa
  _omega = omega or _omega

def reset(nu=None, tau=None, kappa=None, omega=None):
  global nu_slider, tau_slider, kappa_slider, omega_slider
  if nu:
    nu_slider.value = nu
  if tau:
    tau_slider.value = tau
  if kappa:
    kappa_slider.value = kappa
  if omega:
    omega_slider.value = omega

def update_and_render(nu=None, tau=None, kappa=None, omega=None):
  update(nu, tau, kappa, omega)
  #render()
    
    
def point_cloud(depth, pathS):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.
    
    """
    os.chdir(pathS + '/sequence')
    myFiles = glob.glob('*.txt')
    camera_pose = open(pathS + '/sequence/' + '_info.txt','r')
    lines = camera_pose.readlines()


    string =  lines[7]
    t = string.find('=')
    string1 = string[t+2:]

    fx = np.float(string1[:string1.find(' ')])
    string2 = string1[string1.find(' ')+1:]
    string3 = string2[string2.find(' ')+1:]
    cx = np.float(string3[:string3.find(' ')])
    string4 = string3[string3.find(' ')+1:]
    string5 = string4[4:]
    fy = np.float(string5[:string5.find(' ')])
    string6 = string5[string5.find(' ')+1:]
    cy = np.float(string6[:string6.find(' ')])
    cx = 200
    cy = 100
    fx = 200
    fy = 200
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 255)
    z = np.where(valid, depth , np.nan)
    x = np.where(valid, z * (c - cx) / fx, 0)
    y = np.where(valid, z * (r - cy) / fy, 0)
    return np.dstack((x, y, z))



def compute(pathS,pathR):

    os.chdir(pathS + '/sequence')
    myFiles = glob.glob('*.txt')
    camera_pose = open(pathS + '/sequence/'  + '_info.txt','r')
    lines = camera_pose.readlines()
    string =  lines[7]
    t = string.find('=')
    string1 = string[t+2:]

    #877.5 0 479.75 0 0 877.5 269.75

    fx = np.float(string1[:string1.find(' ')])
    string2 = string1[string1.find(' ')+1:]
    string3 = string2[string2.find(' ')+1:]
    cx = np.float(string3[:string3.find(' ')])
    string4 = string3[string3.find(' ')+1:]
    string5 = string4[4:]
    fy = np.float(string5[:string5.find(' ')])
    string6 = string5[string5.find(' ')+1:]
    cy = np.float(string6[:string6.find(' ')])

    #camera to world
    cv2gl = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1],
    ])
    fuze_trimesh = trimesh.load(pathS + '/mesh.refined.obj')

    fuze_trimesh1 = trimesh.load(pathR +  '/mesh.refined.align.trimesh.obj')
    finalPoints_all = np.empty((0,3))
    #for i in range(0,10):
    for i in range (0,len(myFiles)-1):

        if myFiles[i]!='_info.txt':
            #rint (myFiles[i])
            transform_mat = np.loadtxt(pathS+ '/sequence/'+ str(myFiles[i]))
            #print (transform_mat)

        #world to camera
            transform_mat = np.linalg.inv(transform_mat)
            tranform_mat = np.matmul(cv2gl,transform_mat)
            transformation_mat = tranform_mat.transpose()

            fuze_trimesh.apply_transform(tranform_mat)
            mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
            scene = pyrender.Scene()
            scene.add(mesh, pose=np.eye(4))
            cx = 200
            cy = 100
            fx = 200
            fy = 200
            camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy, 0.01, 1000) 
            camera_pose = np.eye(4)
            #camera_pose[2, 3] = 0.1
            scene.add(camera, pose=camera_pose)
            #pyrender.Viewer(scene, use_raymond_lighting=True)


            imgDimX = 400
            imgDimY = 200
            r = pyrender.OffscreenRenderer(imgDimX, imgDimY)
            color, depth = r.render(scene)

          
            fuze_trimesh1.apply_transform(tranform_mat)
            mesh1 = pyrender.Mesh.from_trimesh(fuze_trimesh1)
            scene1 = pyrender.Scene()
            scene1.add(mesh1, pose=np.eye(4))
            scene1.add(camera, pose=camera_pose)
            color1, depth1 = r.render(scene1)

            #change here and select closer depth
            newDepth = depth-depth1
            newDepthN = newDepth
            newDepthN[newDepth<0.10] = 0
            newDepthN[newDepth == depth] = 0
            newDepthN[newDepth == -depth1] = 0
            newDepthF = np.absolute(newDepthN)


            path = pathR + '/'
            my_dpi =  10
            #fig = plt.figure(frameon=False)
            #fig = plt.figure(figsize=(imgDimX/my_dpi, imgDimY/my_dpi), dpi=my_dpi)
            plt.imsave(path  + str(i) + 'tt_nn11.png',newDepthF, cmap=plt.cm.gray_r)



            byteImgIO = io.BytesIO()
            byteImg = Image.open(path + str(i) + 'tt_nn11.png')
            byteImg.save(byteImgIO, "PNG")
            byteImgIO.seek(0)
            byteImg = byteImgIO.read()
            im = np.array(Image.open(io.BytesIO(byteImg)))


            # Precompute a histogram and some integrals.
            n, x, im_bw = im2hist(im)
            prelim = preliminaries(n, x)


            default_nu = np.sum(n)
            default_tau = np.sqrt(1/12)
            default_kappa = np.sum(n)
           # default_omega = 0.50
            default_omega = 0

            _nu = default_nu
            _tau = default_tau
            _kappa = default_kappa
            _omega = default_omega

            #nu=1.00e+9
            #tau=1.58e+7
            omega = 0
            tau = 500e+2
            nu = 500e+2
            kappa = 0
            #kappa=1.58e+8
            #t, score = GHT(n, x, _nu, _tau, _kappa, _omega, prelim)
            t, score = GHT(n, x, 1000, 300, 0.00, 0.00, prelim)

            #new3 400/200/700/700/0/0
            #new 4 400/200/1000/1000/0/0

            F = np.where(im_bw[:,:] < t)

            mask = (im_bw < t)


            depthC = np.minimum(depth, depth1)

            new_array =  a = np.empty((imgDimY,imgDimX))
            new_array[:] = 0
            new_array[mask] = depthC[mask]
            #render(i)


            #fig = plt.figure(frameon=False)
            #fig = plt.figure(figsize=(imgDimX/my_dpi, imgDimY/my_dpi), dpi=my_dpi)
            plt.imsave(path + str(i) + '.png',im_bw < t, cmap=plt.cm.gray_r)



            point_cloudN = point_cloud(new_array,pathS)
            pointsA = point_cloudN.reshape(imgDimY*imgDimX,3)
            #print (pointsA.shape)


            finalPointsFF = pointsA[~np.isnan(pointsA).any(axis=1)]
            b = np.ones((imgDimY*imgDimX,1))
            finalPoints = np.hstack((pointsA,b))

            transform_mat = np.loadtxt(pathS + '/sequence/'+ str(myFiles[i]))
            finalPoints = trimesh.transformations.transform_points(pointsA,transform_mat)
            finalPoints = finalPoints[~np.isnan(finalPoints).any(axis=1)]

            if np.shape(finalPoints)[0] > 0:
                finalPoints_all = np.vstack((finalPoints_all,finalPoints))
            
    np.savetxt(path +'/reprojected_10.xyz', finalPoints_all, delimiter=' ') 


if __name__ == '__main__':
    file1 = open('validation-scan-rescan.txt')
    Lines = file1.readlines()

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
                    print (dd)                        
                    compute(pathS,pathR)
                    print ("rescan ",pathR," is completed") 

