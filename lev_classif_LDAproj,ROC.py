#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Levitation height distribution extraction analysis
#Last updated 4 March 2021.
#Author: Shreya Deshmukh, Stanford University.

#Runs as a jupyter notebook through anaconda (on python version 2.7)

#Import the following libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, roc_curve, auc

plt.rc('font',family='Arial')

#Define file path
#path = 'C:\\Users\\yourefiledirectories' #input the file path to where statistical metrics of height distributions are stored


# In[ ]:


#Load PBS (low medium density protocol) data, statistical metrics of all samples' height distributions
df = pd.read_csv('C:\\yourfiledirectories\\statmoments.csv', sep=',', names =
               ['labels','Mean','Variance','Skewness','Kurtosis'])

X = df.iloc[:,1:5].copy() #features
y = df.iloc[:,0].copy() #labels
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0) #split into training and test sets

#Train Linear Discriminant Analysis (LDA) on training data
LDA = LinearDiscriminantAnalysis(n_components=1) # projection in 1D space
data_proj = LDA.fit_transform(X_train,y_train)

#Put scores and truth labels in same array
clsfd = np.zeros((data_proj.shape[0],2))
clsfd[:,0]=data_proj[:,0]
clsfd[:,1]=y_train

#Apply LDA projection from training data to test data
data_test = LDA.fit_transform(X_test,y_test)
clsfdtest = np.zeros((data_test.shape[0],2))
clsfdtest[:,0]=data_test[:,0]
clsfdtest[:,1]=y_test

#Plot data
plt.figure(1)
plt.figure(figsize=(6,2))
plt.scatter(clsfd[:,0],clsfd[:,1], marker = "^", color = 'red', label ='Train')
plt.scatter(clsfdtest[:,0],clsfdtest[:,1], color = 'blue', label ='Test')
#plt.ylim([-1,2])
plt.axis([-4, 4, -1, 2])
plt.xlabel('LDA score')
plt.yticks=([0,1],('Healthy','Infected'))
plt.legend(loc = 'lower right')


# In[ ]:


#Load Percoll (high medium density protocol) data, statistical metrics of all samples' height distributions
dfP = pd.read_csv('C:\\Users\\17742\\Documents\\Malaria project\\Data\\LDAdata_Perc.csv', sep=',', names =
               ['labels','Mean','Variance','Skewness','Kurtosis'])
XP = dfP.iloc[:,1:5].copy() #features
yP = dfP.iloc[:,0].copy() #labels
X_trainP, X_testP, y_trainP, y_testP = train_test_split(XP,yP,test_size=0.3,random_state=0) #split dataset into training and test

#Train LDA on training set
LDA = LinearDiscriminantAnalysis(n_components=1) # projection in 1D space
data_projP = LDA.fit_transform(X_trainP,y_trainP)
data_testP = LDA.fit_transform(X_testP,y_testP) #apply projection to test set

#Put scores and labels in the same array
clsfdP = np.zeros((data_projP.shape[0],2))
clsfdP[:,0]=data_projP[:,0]
clsfdP[:,1]=y_trainP

clsfdtestP = np.zeros((data_testP.shape[0],2))
clsfdtestP[:,0]=data_testP[:,0]
clsfdtestP[:,1]=y_testP

#Plot data
plt.figure(2)
plt.figure(figsize=(6,2))
plt.scatter(clsfdP[:,0],clsfdP[:,1], marker = 'o', color = 'red', label ='Train')
plt.scatter(clsfdtestP[:,0],clsfdtestP[:,1], marker = '^', color = 'blue', label ='Test')
#plt.ylim=([-2, 3])
plt.axis([-4, 4, -1, 2])
plt.xlabel('LDA score')
plt.yticks=([0,1],('Healthy','Infected')) 
plt.legend(loc = 'lower right')


# In[ ]:


#Organise data derived from LDA projections by their "ground truth" labels
PBSmat = np.concatenate((clsfd, clsfdtest), axis=0) #concatenate training and test sets
PBSssort = PBSmat[PBSmat[:,0].argsort()] #sort by LDA "scores"
PBSlsort = PBSmat[PBSmat[:,1].argsort()] #sort by healthy/infected labels

Percmat = np.concatenate((clsfdP, clsfdtestP), axis=0)
Percssort = Percmat[Percmat[:,0].argsort()]
Perclsort = Percmat[Percmat[:,1].argsort()]


#Identifying index where 0s (healthy labels) split from 1s (infected labels) in label-sorted data list
#PBS/low medium density
if PBSlsort[0,1] == 0 and PBSlsort[PBSlsort.shape[0]-1,1] == 1:
    for i in range(1,PBSlsort.shape[0]):
        if PBSlsort[i,1] != PBSlsort[i-1,1]:
            startinfpbs = i

mean_hPBS = np.mean(PBSlsort[0:startinfpbs-1,0])
mean_iPBS = np.mean(PBSlsort[startinfpbs:PBSlsort.shape[0],0])
thresh_PBS = np.mean([mean_hPBS,mean_iPBS])

#Percoll/high medium density
if Perclsort[0,1] == 0 and Perclsort[Perclsort.shape[0]-1,1] == 1:
    for i in range(1,Perclsort.shape[0]):
        if Perclsort[i,1] != Perclsort[i-1,1]:
            startinf = i

mean_hPerc = np.mean(Perclsort[0:startinf-1,0])
mean_iPerc = np.mean(Perclsort[startinf:Perclsort.shape[0],0])
thresh_Perc = np.mean([mean_hPerc,mean_iPerc])

#Split data into healthy (uninfected) and infected arrays
data_hPerc = Perclsort[0:startinf-1,0]
data_iPerc = Perclsort[startinf:Perclsort.shape[0],0]
data_hPBS = PBSlsort[0:startinfpbs-1,0]
data_iPBS = PBSlsort[startinfpbs:PBSlsort.shape[0],0]


# In[ ]:


#Plot LDA score results for uninfected vs. infected

#Load PBS (low medium density protocol) data
XPBS = np.loadtxt('C:\\yourfiledirectories\\LDAscores.csv', delimiter=',', encoding='utf-8-sig')

#Load Percoll (high medium density protocol) data
XPerc = np.loadtxt('C:\\yourfiledirectories\\LDAscores_Perc.csv', delimiter=',', encoding='utf-8-sig')

n = 1

smallmarker = dict(markersize=3)

data_hPBS = np.array(XPBS[0:11,n])
data_iPBS = XPBS[12:31,n]
data_hPerc= XPerc[0:11,n]
data_iPerc = XPerc[12:19,n]


#Making box and whisker plots #PBS/low density medium
plt.figure(1)
plt.figure(figsize=(2,3))
#plt.title('High-density medium', fontsize=15, fontname='Arial')
boxPerc = plt.boxplot([data_hPerc, data_iPerc], labels=('Uninfected','Infected'), widths=0.5, flierprops = None, patch_artist=True)
plt.ylabel('LDA score', fontname='Arial', fontsize = 13)
# fill with colors
colors = ['cornflowerblue', 'white']
for patch, color in zip(boxPerc['boxes'], colors):
    patch.set_facecolor(color)


#Making box and whisker plots #Percoll/high density medium
plt.figure(2)
plt.figure(figsize=(2,3))
#plt.title('High-density medium', fontsize=15, fontname='Arial')
boxPBS = plt.boxplot([data_hPBS, data_iPBS], labels=('Uninfected','Infected'), widths=0.5, flierprops = None, patch_artist=True)
plt.ylabel('LDA score', fontname='Arial', fontsize = 13)
# fill with colors
colors = ['sandybrown', 'white']
for patch, color in zip(boxPBS['boxes'], colors):
    patch.set_facecolor(color)


# In[ ]:


#Apply ROC (receiver operating characteristic) analysis to LDA data to evaluate classification thresholds
#PBS/low density medium protocol

data_truth = PBSmat
no_splits = data_truth.shape[0]
ROCmat = np.zeros((10,no_splits+1))
start = np.min(data_truth[:,0]) #mean_hPerc
end = np.max(data_truth[:,0]) #mean_iPerc

for t in range(0,ROCmat.shape[1]):
    thresh = start + (1./float(no_splits))*t*(end-start)
    ROCmat[0,t] = thresh
    pred = np.zeros((data_truth.shape[0],1))
    for i in range(0,data_truth.shape[0]):
        if data_truth[i,0] < thresh:
            pred[i,0] = 0
        elif data_truth[i,0] >= thresh:
            pred[i,0] = 1
    tn, fp, fn, tp = confusion_matrix(data_truth[:,1],pred[:,0]).ravel()
    ROCmat[1:5,t] = (tn, fp, fn, tp)
    p = fn + tp #no. of real positive cases
    n = tn + fp #no. of real negative cases
    ROCmat[5,t] = float(fp)/n #fpr, false positive rate, x axis of ROC
    ROCmat[6,t] = float(tp)/p #tpr, true positive rate, sensitivity, y axis of ROC
    ROCmat[7,t] = float(tn)/n #tnr, true negative rate, specificity
    ROCmat[8,t] = float(tp + tn)/(p + n) #accuracy

ind = np.argmax(ROCmat[6,:]-ROCmat[5,:])
threshPBS = ROCmat[0,ind]

ROCPBS=ROCmat
aucPBS = auc(ROCPBS[5,:],ROCPBS[6,:])

#Plotting final ROC curve
plt.figure(6)
plt.figure(figsize=(4,4))
#plt.axis([-0.05, 1.05, -0.05, 1.05])
plt.plot(ROCmat[5,:], ROCmat[6,:])
plt.plot(range(0,ROCmat.shape[1]),range(0,ROCmat.shape[1]), "--")
plt.grid([])
plt.legend(["AUC=%.3f"%aucPBS], loc = 'lower right')

print(threshPBS)
print(ROCmat[1:9,ind])
print(ROCmat[4,ind])


# In[ ]:


#Apply ROC (receiver operating characteristic) analysis to LDA data to evaluate classification thresholds
#Percoll/high density medium protocol

data_truth = Percmat
no_splits = data_truth.shape[0]
ROCmat = np.zeros((10,no_splits+1))
start = np.min(data_truth[:,0]) #mean_hPerc
end = np.max(data_truth[:,0]) #mean_iPerc

for t in range(0,ROCmat.shape[1]):
    thresh = start + (1./float(no_splits))*t*(end-start)
    ROCmat[0,t] = thresh
    pred = np.zeros((data_truth.shape[0],1))
    for i in range(0,data_truth.shape[0]):
        if data_truth[i,0] < thresh:
            pred[i,0] = 0
        elif data_truth[i,0] >= thresh:
            pred[i,0] = 1
    tn, fp, fn, tp = confusion_matrix(data_truth[:,1],pred[:,0]).ravel()
    ROCmat[1:5,t] = (tn, fp, fn, tp)
    p = fn + tp #no. of real positive cases
    n = tn + fp #no. of real negative cases
    ROCmat[5,t] = float(fp)/n #fpr, false positive rate, x axis of ROC
    ROCmat[6,t] = float(tp)/p #tpr, true positive rate, sensitivity, y axis of ROC
    ROCmat[7,t] = float(tn)/n #tnr, true negative rate, specificity
    ROCmat[8,t] = float(tp + tn)/(p + n) #accuracy

ind = np.argmax(ROCmat[6,:]-ROCmat[5,:])
threshPerc = ROCmat[0,ind]

ROCPerc=ROCmat
aucPerc = auc(ROCPerc[5,:],ROCPerc[6,:])

#Plotting final ROC curve
plt.figure(5)
plt.figure(figsize=(4,4))
plt.axis=([-0.05, 1.05, -0.05, 1.05])
plt.plot(ROCmat[5,:], ROCmat[6,:])
plt.plot(range(0,ROCmat.shape[1]),range(0,ROCmat.shape[1]), "--")
plt.grid([])
plt.legend(["AUC=%.3f"%aucPerc])

print(threshPerc)
print(ROCmat[1:9,ind])


# In[ ]:


#Plotting ROCs on top of each other with proper scaling
#reload(plt)
plt.figure(7)
plt.figure(figsize=(4,4))
#plt.axis([-0.05, 1.05, -0.05, 1.05])
plt.ylim((-0.05,1.05))
plt.xlim((-0.05,1.05))
plt.plot(ROCPBS[5,:], ROCPBS[6,:], color = 'sandybrown', label = "low density AUC=%.3f"%aucPBS)
plt.plot(ROCPerc[5,:], ROCPerc[6,:], color = 'cornflowerblue', label = "high density AUC=%.3f"%aucPerc)
plt.plot(range(0,ROCmat.shape[1]),range(0,ROCmat.shape[1]), "--", color = 'black')
plt.xlabel('False positive rate', fontname='Arial', fontsize=13)
plt.ylabel('True positive rate', fontname='Arial', fontsize=13)
plt.legend(loc = 'lower right')

