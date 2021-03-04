# levitate-RBC

Author: Shreya Deshmukh (Stanford University).
Last updated 4 March 2021.

This code is written for the purpose of modelling the biophysical manipulation of cells in a microscale magnetic levitation system, and measuring from image data the results of levitation as height distributions in order to classify their malaria infection status. This work has been developed in the Demirci research group at Stanford University's Canary Center for Early Cancer Detection.

System requirements:
This software has been tested on a 64-bit machine with a Windows 10 operating system. It does not require any non-standard hardware.
This code has been run as part of the "jupyter notebook" implementation of python 2.7 through Anaconda, using the .ipynb files that can be found in this repository to process data of cell levitation images. It also depends on the installation (e.g. through pip install) of a number of modules, including: OpenCV for python ("cv2", up to version 4.2.0.32 is supported for python 2.7). 
Note: python 2.7 is reaching deprecation in favour of python 3, and while this code can be used as-is with python 2.7, it can also be used with python 3 with minimal syntax adaptations.

Installation instructions: (typical download time for these files should be less than 1 minute)
1. Ensure that dependencies are installed as described above. Download files, and if using as part of jupyter notebook, open jupyter notebook through Anaconda then open the .ipynb files.
2. The "levmodels-eqns,sims" file can be used to model predicted levitation heights of different cell types with varying device and protocol parameters. It can be run on its own using the cell parameters already part of the code, but these can be modified to test with the user's own cell parameters. The latter portion of this code can be used to simulate the distribution of cell heights based on the predicted heights generated in the earlier portion of the code. Expected runtime is <1 second on a typical desktop computer.
3. The "lev_extractheights" file can be used to process image data to extract the height distributions of cells in levitation. The output is an array of values of cell-pixel density by height across the device channel. Within the code are functions to calculate the statistical moments for the height distribution (mean, variance, skewness, kurtosis), which are the values used to calculate Linear Discrimnant Analysis (LDA) scores in the next step. Expected runtime per image is <30 seconds on a typical desktop computer.
4. The "lev_classif_LDAproj,ROC" file can be used to calculate LDA projections from the statistical moments data (from the previous step), using training data to generate projections that can be applied to the test data, and the resulting LDA scores can be used to classify samples as malaria-infected or uninfected using a threshold. The latter portion of this code calculates receiver operating characteristic (ROC) curves for the LDA data. Expected runtime is <60 seconds on a typical desktop computer.

The code can be tested on the associated demo example data available with the files. 
