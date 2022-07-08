# ObjectsCanMove
# Objects can move: Change Detection by Geometric Transformation Consistency

This is the code for the evaluation of our results on the [3Rscan dataset](https://arxiv.org/pdf/1908.06109.pdf), a dataset built towards object instance relocalization. The pipeline discovers objects based on change detection. Evaluation is performed in terms of Accuracy, Completeness, Recall and IoU.


## Setting up environment
* Download and extract the validation split of the dataset from [here](https://waldjohannau.github.io/RIO/). This dataset has been compiled for object instance relocalization, hence we provide the script `create_GTchanges.py` for computing changes between the reference and the rescan, based on the JSON file provided with the dataset and the ground-truth annotations.

* Download and store all our provided scripts in the main folder of the dataset (i.e. a folder entitled `3Rscan` dataset, containing all the subfolders of the scenes). Please also store in this folder `3RScan.json`, as provided by 3RScan contributors. We are also providing `environment.yml`, with the dependencies needed to execute our scripts.

* If you are using conda, run:
``` bash
conda env create -f environment.yml
```

## Generate ground-truth annotations for change detection:

* To generate the 3D objects that have been moved according to files provided by the 3RScan dataset, you have to run the script `create_GTchanges.py`. To be able to execute the script, align each segmentation (i.e the file `labels.instances.annotated.ply`) for each rescan to the reference scan, as described in [here](https://waldjohannau.github.io/RIO/). Name each aligned rescan as `labels.instances.annotated.align.ply`. Now, each folder containing a rescan should contain `labels.instances.annotated.align.ply` and `labels.instances.annotated.ply`.

* Once you have wrapped the scans, execute the command:
```bash
python create_GTchanges.py
```

* This script  will create for each folder the file `groundTruthChanged_withoutaddedobjects.xyz`. If you want to include in the file objects that have also been added, uncomment the lines of code, according to documentation.

## Preprocessing:


* Our approach assumes that the scans are aligned to each other. Wrap each rescan to the reference scan (.obj files), as dictated by [here](https://waldjohannau.github.io/RIO/). Name each aligned rescan as `mesh.refined.align.trimesh.obj`. Each folder containing a rescan should now include `mesh.refined.obj` and `mesh.refined.align.trimesh.obj`.


* Compute the normals for each .obj file by executing at the command line:
```bash
python convertToPCD.py
```

* Compute the supervoxels using [PCL](https://pointclouds.org/documentation/tutorials/supervoxel_clustering.html). Build the `supervoxel_clustering.cpp` using the`CMakeLists.txt` provided. After building the script, execute at the command line:
```bash
./supervoxel_clustering --NT
```

 * This will store the a .pcd with the created supervoxels for all the folders (containing reference scans and rescans). It will also create a .txt file, storing for each supervoxel, its adjacent supervoxels.


## Run object discovery pipeline:

* Computed initial changed regions (from depth map rendering, subtraction, and thresholding). Execute at the command line:
```bash
python initialDetection.py
```
* This will creat a file `reprojected_10.xyz`  containing all the initial detections for each rescan.


* Compute DGNN features, using [DCCNN pre-trained model]( https://github.com/AnTao97/dgcnn.pytorch).  Every kind of features (contrastive, learned, or handcrafted) can be used, as long as they return a .npy file containing the points and a corresponding .npy file containing the features.



* Given the extracted points and features in the form of files `points.npy`, `features.npy` respectively, running `computeMatches.py` establishes correspondences between the reference scan and the rescan. For better computational performance, [FAISS](https://github.com/facebookresearch/faiss) is used. Make sure it is installed in your system. To compute matches execute at the command line:
```bash
python computeMatches.py
```
* Compute transformations with `computeConsistentTransformations.py`. This will create in each subfolder of the rescans, a new folder, storing all the transformations, and the consistent points for each transformation, both for the reference scan and the rescan. Run the script:
 ```bash
 python computeConsistentTransformations.py
```

* Perform graph optimization.  Download and install [Min-cut Pursuit](https://gitlab.com/1a7r0ch3/parallel-cut-pursuit).
Run the script by executing at the command prompt:
 ```bash
python prepareVariablesForCutPursuit.py
```
* This will prepare and store all the variables needed for the graph optimization step. Then, execute at the command line:
```bash
python SolveCutPursuit.py
```
* Make sure that you have installed all the appropriate files and that the script has access to the python wrapper. The execution of this script will create a file entitled `final_changing_points.npy`, with all the points labeled as changing from our procedure.

## Evaluation

For the computation of the metrics 4 files are available:
*  Compute mean IoU (voxel level) - execute at the command line:
 ```bash
python meanIoU.py
```
*  Compute mean recall (voxel level) - execute at the command line:
```bash
python meanRecall.py
```
*  Compute mean accuracy -  mean completeness (point level). To run this script you must compute the mean Iou before. Execute at the command line:
```bash
python meanAccComplet.py
```

[**WARNING!**] The results are hard-coded to be reproducible. In the future release of the code different hyperparameters will be able to be tuned as arguments of the relevant scripts. Even though the parameters are tuned for our experiments,  an element of randomness remains in the result due to RANSAC execution.
In this distribution, we provide scripts for our core method, as well as the baseline of Palazzolo et al. Palazzolo et al. is equivalent to our method before optimization, so stopping after initial detection, reproduces the output of this baseline. Replacing the computed transformations by the ground-truth transformations provided by 3RScan dataset, during the graph optimization step will reproduce the ablation baseline of ground-truth transforms. 

We decided to distribute this version in parts, so as to be easier to reproduce some baseline experiments. If you have completed the preprocessing and computed DGCNN features, you can run the whole pipeline using `objectsCanMove.sh` by executing at the command line:
 ```bash
bash objectsCanMove.sh
```


### Paper
If you find the data useful please consider citing our [paper]:

```
Objects Can Move: 3D Change Detection by Geometric Transformation Consistency, A. Adam, K.Karantzalos, T.Sattler, T.Pajdla (to appear at ECCV'22)
```
