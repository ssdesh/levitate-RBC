#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Levitation height distribution extraction analysis
#Last updated 4 March 2021.
#Author: Shreya Deshmukh, Stanford University.

#Runs as a jupyter notebook through anaconda (on python version 2.7)

#Import the following libraries
from os import listdir
from os.path import join
import numpy as np
import scipy.misc
from scipy import ndimage

from scipy import stats
from scipy.stats import moment, mode
import csv
import matplotlib.pyplot as plt
import cv2
import os.path
from PIL import Image
import skimage
from skimage.transform import resize
from scipy.signal import savgol_filter
get_ipython().run_line_magic('matplotlib', 'inline')

#Define file path
#path = 'C:\\Users\\yourefiledirectories' #input the file path to where images are stored in the directory


# In[ ]:


img_title = "Snap-153_c1"

#Import image as RGB
img_bf = cv2.imread(os.path.join(path, img_title + ".tif")) 

#Convert from RGB to grey using OpenCV inbuilt command -brightfield
grey = cv2.cvtColor(img_bf, cv2.COLOR_BGR2GRAY)
plt.figure(figsize = (4,4))
plt.imshow(grey, cmap = plt.get_cmap('gray'))


# In[ ]:


#Rotate image to parallel using scipy ndimage if needed (0 if already parallel)
rot_ang = 0 #degrees of rotation

rot = ndimage.rotate(grey, rot_ang)
plt.figure(figsize = (8,8))
plt.imshow(rot, cmap = plt.get_cmap('gray'))


# In[ ]:


#Crop image to channel dimensions (removing extra space on top and bottom)
#Crop vertically using user-selected channel wall coordinates (depending on how the device was positioned for that image)

cropr_t = 80 #top
cropr_b = 740 #bottom; keep a difference of 660
cropc_l = 200 #left
cropc_r = 1200 #right
        
crop = rot[cropr_t:cropr_b,cropc_l:cropc_r]

#Stretch factor (double pixels vertically and horizontally) applied to old microscope's images due to their smaller dimensions
resized = skimage.transform.resize(crop, (int((cropr_b-cropr_t)*1.25),1267)) #for old microscope
#resized = skimage.transform.resize(crop, (int((cropr_b-cropr_t)*0.925),1689)) #for new microscope

#Pixels in the image approximately correspond to microns in the device dimensions after image adjustment to account for different microscope cameras.

plt.figure(figsize = (10,10))
plt.imshow(resized, cmap = plt.get_cmap('gray'))


# In[ ]:


#Apply Gaussian smoothing 
blur = cv2.GaussianBlur(resized, (3,3), 2.7, 2.7)

from scipy.misc import bytescale
blur8 = bytescale(blur)
plt.figure(figsize = (8,8))
plt.imshow(blur8, cmap = plt.get_cmap('gray')) 

#Apply Laplacian filter 
smooth  = cv2.Laplacian(blur8, 8)
plt.figure(figsize = (8,8))
plt.imshow(smooth, cmap = plt.get_cmap('gray')) 

#Binarise using Otsu's thresholding -brightfield
ret,th = cv2.threshold(smooth,0,1,cv2.THRESH_OTSU) #with Otsu's method, pos 2 is an arbitrary float (default: 0), pos 3 is the max value desired in output (1), pos 4 selects the thresholding fcn
#print ret

plt.figure(figsize = (12,12))
plt.imshow(th, cmap = plt.get_cmap('gray')) 

cropper=5 #remove edge of image where binarisation adds a contrasting border artifact
plt.figure(figsize = (12,12))
plt.imshow(th[0:th.shape[0]-cropper,0:th.shape[1]-cropper], cmap = plt.get_cmap('gray'))


# In[ ]:


#Sum intensity values for each row (across the columns)

#BRIGHTFIELD
n_rowsb = th.shape[0]-cropper
intmatb = np.empty((n_rowsb,1))
for i in range(0,n_rowsb):
    intmatb[i] = sum(th[i,0:th.shape[1]-cropper])
    
#Background noise subtraction
p_intmat = intmatb

p_intmat[0:2]=0 #removing border artifact

#Pad 100 on bottom, and approx. 100 on top (very bottom and top of the capillary is obscured due to refraction at the glass interfact)
p_intmatpad = np.pad(p_intmat, [(0, int(0.5*(1000-th.shape[0]))+cropper), (0, 0)], mode='constant', constant_values=0)

rev_intmatb = p_intmatpad[::-1]

#Pad top with zeros (approx. 100) to finalise array height to 1000
xp_bf = np.pad(rev_intmatb, [(0, 1000-rev_intmatb.shape[0]), (0, 0)], mode='constant', constant_values=0)


# In[ ]:


#Plot the height distribution (sanity check)
plt.figure(figsize=(4,4))
plt.plot(range(0,xp_bf.shape[0]), xp_bf, markersize=2, label = "Brightfield")
plt.legend()

plt.ylabel('Number of cell-positive pixels', fontsize=18)
plt.xlabel('Levitation height (${\mu}m$)', fontsize=18)


# In[ ]:


print sum(xp_bf)

#Prepare height distribution for extraction of statistical moments
hts = xp_bf 

#Compute statistical metrics for the distribution (mean, variance, skewness, kurtosis)
data_arr = hts

#Statistical analysis of heights
multilist = [None]*data_arr.shape[1] #prepare empty list

#Loop through each height distribution
for i in range(0,data_arr.shape[1]):
    hlist = []
    count = 1
    heights = data_arr[:,i]
    for h in heights:
        hlist = hlist + [count]*int(h)
        count = count+1
    multilist[i]=hlist
    
#Store statistical descriptors of these height distributions: columns are different sample types;
#Rows: mean, variance, skewness, kurtosis, standard deviation, median, mode
multistat = np.empty([4,data_arr.shape[1]])
for i in range(0,data_arr.shape[1]):
    multistat[:,i] = [np.mean(multilist[i]), np.var(multilist[i]), moment(multilist[i],moment=3), moment(multilist[i],moment=4)] #, np.sqrt(np.var(jlist)), np.median(jlist)]

print multistat


# In[ ]:


# Plot the brightfield cell height distribution

#Smoothing with Savitzky-Golay filter
hts_smooth = savgol_filter(hts[:,0], 87, 4) # window size, polynomial order

#Plot smoothed curve on top of raw data
plt.figure(figsize=(6,9))
plt.plot(hts[:,0], range(0,1000), markersize=2, label = "Brightfield")
plt.plot(hts_smooth, range(0,1000), markersize=2, label = "Brightfield, smoothed")
plt.xlabel('Number of cell-positive pixels', fontsize=15)
plt.ylabel('Levitation height (${\mu}m$)', fontsize=15)
plt.legend(fontsize=12)


# In[ ]:


#Save all variables into a csv file
stack = np.column_stack((hts1, hts2, hts3, hts4))
np.savetxt('yourfilename.csv', stack, delimiter = ",") 


# In[ ]:


#Load saved data from CSV file
hts_load = np.array(list(csv.reader(open("yourfilename.csv", "rb"), delimiter=","))).astype("float")

