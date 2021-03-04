#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Levitation height modelling analysis
#Last updated 3 March 2021.
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
#import cv2
from PIL import Image
import math
get_ipython().run_line_magic('matplotlib', 'inline')

#Define file path
#path = 'C:\\Users\\yourefiledirectories'


# #Define medium variables (as an array)
# #Medium magnetic susceptibility
# Gd_vec_pre = range(0,11,1) #molar concentration of gadolinium (paramagnetic) ions
# Gd_vec = [0.01*i for i in Gd_vec_pre] #unit conversion
# 
# #Medium density
# Percoll_vec_pre = range(101,112,1) #density of Percoll solution used (rho_PBS when Gd = 10mM is 1013219.67213)
# Percoll_vec = [0.01*i for i in Percoll_vec_pre] #unit conversion
# 
# #Define universal and device constants
# g = 9.8 #m*s^-2; acceleration of gravity
# p = 0.0125663706 #permeability of free space = 0.01257 m·g·s-2·A-2
# d = 0.001 #height of channel (separation between magnets), in m (0.001 in our device)
# B = 375 #(375 mT) #surface magnetic field strength for the NdFeB magnets we used, in mT, or g·s-2·A-1
# rho_H2O = 10**6 #density of water at room temperature
# chi_m_Gd = 0.000334 #molar magnetic susceptibility of gadolinium ions
# chi_H2O = -0.719*10**(-6) #molar magnetic susceptibility of water at room temperature
# chi_factor = 4*math.pi #conversion factor for (unitless) magnetic susceptibility
# 
# #THIS FAR
# 
# #rho_PBS = ((1.01*(1-Gd))+(1.3*(Gd/0.976)))*np.power(10,6) #dependent on Gd
# #rho_Percoll = ((Percoll*(1-Gd))+(1.3*(Gd/0.976)))*np.power(10,6) #dependent on Gd and Percoll
# 
# #rho_medium = rho_Percoll;  #dependent on Gd and/or Percoll
# 

# In[ ]:


#Calculating predicted levitation height of a cell with input parameters

#Define medium variables (as an array)
#Medium magnetic susceptibility
Gd_vec_pre = range(0,11,1) #molar concentration of gadolinium (paramagnetic) ions
Gd_vec = [0.01*i for i in Gd_vec_pre] #unit conversion

#Medium density
Percoll_vec_pre = range(101,112,1) #density of Percoll solution used (rho_PBS when Gd = 10mM is 1013219.67213)
Percoll_vec = [0.01*i for i in Percoll_vec_pre] #unit conversion

#Define universal and device constants, display values to check results
g = 9.8 #in m*s^-2; acceleration of gravity
d = 0.001 #in m, height of channel (separation between magnets), in m (0.001 in our device)
chi_m_Gd = 3.34*10**(-7) #in m^3*mol^-1, molar magnetic susceptibility of gadolinium ions
chi_factor = 4*math.pi #conversion factor for (unitless) magnetic susceptibility
B = 0.375 #in T, surface magnetic field strength for the NdFeB magnets we used, in T, or kg·s-2·A-1
p = 0.00000125663706 #in T*m*A^-2, permeability of free space
Gd = 0.03 #in mol*m^-3, molar concentration of gadolinium (paramagnetic) ions
rho_cell = 1.1*10**3 #in kg*m^3, density of the RBC
chi_cell = -5*10**(-6) #unitless, magnetic susceptibility of the cell
chi_H2O = -0.719*10**(-6) #molar magnetic susceptibility of water at room temperature
rho_medium = 1.005*10**3 #in kg*m^3, density of the medium (PBS)
print rho_medium, "rho_medium"

delta_rho = rho_cell - rho_medium #difference in density between cell and medium
print delta_rho, "delta_rho"
print chi_cell, "chi_cell"

chi_medium_calc = (Gd*chi_m_Gd*1000)-0.000009 #magnetic susceptibility of medium based on the concentration of gadolinium ions used
print chi_medium_calc, "chi_medium_calc"
chi_medium = chi_medium_calc
print chi_medium, "chi_medium"

delta_chi = chi_cell - chi_medium #difference in magnetic susceptibility between cell and medium
print delta_chi, "delta_chi"

#numerator from Eq. 1 in Methods
num = delta_rho*g*p*d**2
print num, "num"
#denominator from Eq. 1 in Methods
denom = delta_chi*4*B**2
print denom, "denom"
print num/denom, "fraction"
h_cell=((0.5*d)+(num/denom))*10**6 #h_cell is the predicted levitation height of the cell under the defined parameters of the system
print h_cell, "h_cell"


# In[ ]:


#Define cell variables

#chi_PBS = chi_H2O + (Gd*chi_m_Gd*chi_factor); #dependent on Gd
chi_WBC = chi_H2O #leukocyte magnetic susceptibility
rho_agran = 1.0795*10**6 #leukocyte (agranulocyte) density
rho_gran = 1.068*10**6 #leukocyte (granulocyte) density

chi_RBC = chi_factor*(chi_H2O+((-0.18)*10**(-6))) #RBC (uninfected) magnetic susceptibility
rho_RBC = 1.100*10**6 #1.125*10**6 #RBC (uninfected) density

chi_troph = chi_factor*(chi_H2O+(0.91*10**(-6))) #infected RBC (trophozoite) magnetic susceptibility
rho_troph =1.079*10**6 #infected RBC (trophozoite) density

chi_ring = chi_factor*(chi_H2O+(0.82*10**(-6)))  #infected RBC (ring) magnetic susceptibility
rho_ring = 1.090*10**6 # #infected RBC (trophozoite) density

chi_schiz = chi_factor*(chi_H2O+(1.80*10**(-6))) # #infected RBC (schizont) magnetic susceptibility
rho_schiz = 1.070*10**6 #1.09*10**6 # #infected RBC (schizont) density


# In[ ]:


#Calculating predicted levitation height of a cell with varying medium parameters (as an array)
#performed with inputs for "RBC" (i.e. uninfected RBCs), but can be replaced with variables for other cell types

#Input variables and constants into equations and show results (select cell type here to generate values specific to that cell type or stage)
chi_cell = chi_RBC
rho_cell = rho_RBC

h_cell_arr = np.empty([len(Gd_vec),len(Percoll_vec)])

for i in range(0,len(Gd_vec)):
    Gd = Gd_vec[i] #this vector is the range of paramagnetic ion concentrations evaluated
    chi_PBS = chi_H2O+(Gd*chi_m_Gd*chi_factor); #dependent on Gd
    delta_chi = chi_cell - chi_PBS;
    for j in range(0,len(Percoll_vec)):
        Percoll = Percoll_vec[j] #this vector is the range of medium densities evaluated
        rho_medium = ((Percoll*(1-Gd))+(1.3*(Gd/0.976)))*10**6
        delta_rho = rho_cell - rho_medium
        #print delta_rho
        h_cell_arr[i,j]=((0.5*d)+((delta_rho*g*p*d**2)/(delta_chi*4*B**2)))*10**6
        
h_cell_RBC = h_cell_arr

#print h_cell_RBC


# In[ ]:


#Calculate theoretical separation between cell types under those specific levitation conditions
sep = np.subtract(h_cell_agran,h_cell_RBC)
max_s = np.amax(sep)
max_sep_ind = np.where(sep==max_s)

print max_s
print Gd_vec[max_sep_ind[0]]
print Percoll_vec[max_sep_ind[1]]


# In[ ]:


#Plot theoretical levitation heights 
plt.figure(1)
plt.figure(figsize=(6,4))
x = [1,2,3,4]
y_lowrho = [h_cell_RBC[0,0],h_cell_ring[0,0],h_cell_troph[0,0],h_cell_schiz[0,0]]
plt.scatter(x,y_lowrho, c = 'r', s=80, marker = 's', label = "In low-density medium")
plt.hold(True)
max_rho = h_cell_RBC.shape[1]
y_highrho = [h_cell_RBC[0,max_rho-1],h_cell_ring[0,max_rho-1],h_cell_troph[0,max_rho-1],h_cell_schiz[0,max_rho-1]]
plt.scatter(x,y_highrho, c = 'r', s=80, marker = 'o', label = "In high-density medium")
plt.xticks(x, ('Uninfected', 'Ring-stage', 'Trophozoite', 'Schizont'), fontsize=10, rotation=0)
plt.xlabel('Erythrocyte type by infection stage', fontsize=12)
plt.ylabel('Channel height, z (${\mu}m$)', fontsize=12)
plt.legend(loc='lower right', fontsize=11)


# In[ ]:


#Plot theoretical predicted values for different cells' levitation heights and their separation over a range of levitation conditions (varying medium density, and then varying medium magnetic susceptibility)

plt.figure(2)
plt.figure(figsize=(4.5,4.5))
x = 0.01*np.arange(101,112)
y_RBC = h_cell_RBC[0,:]
y_ring = h_cell_ring[0,:]
y_RBC40 = h_cell_RBC[1,:]
y_ring40 = h_cell_ring[1,:]
y = y_ring - y_RBC
plt.scatter(x,y, label = "$40mM$") #, c = 'bl', s=100, marker = 'o')

#plt.hold(True)
y40 = y_ring40 - y_RBC40
plt.axis([1.00, 1.12, 0, 27], fontsize=18)
plt.xlabel('Medium density (g/mL)', fontsize=12)
plt.ylabel('Separation in levitation height (${\mu}m$)', fontsize=12)

plt.figure(3)
plt.figure(figsize=(4.5,4.5))
Gd = 0.01*np.arange(3,14)
x = pow(10,4)*(((-9.05)*pow(10,-6))+(Gd*0.00032*(9.05/0.719)))
y_RBC = h_cell_RBC[:,0]
y_ring = h_cell_ring[:,0]
y = y_ring - y_RBC

max_chi = h_cell_RBC.shape[0]
y_RBCdense = h_cell_RBC[:,max_chi-1]
y_ringdense = h_cell_ring[:,max_chi-1]
ydense = y_ringdense - y_RBCdense
plt.scatter(x,ydense, label = "high-density") #, c = 'bl', s=100, marker = 'o')
plt.axis([0.75, 5.5, 0, 27], fontsize=18)
plt.xlabel('Medium magnetic susceptibility (x$10^4$)', fontsize=12)
plt.ylabel('Separation in levitation height (${\mu}m$)', fontsize=12)


# In[ ]:


plt.figure(2)
plt.figure(figsize=(4.5,4.5))
x = 0.01*np.arange(101,112)
y_RBC = h_cell_RBC[0,:]
y_ring = h_cell_ring[0,:]
y_RBC40 = h_cell_RBC[1,:]
y_ring40 = h_cell_ring[1,:]
y = y_ring - y_RBC
plt.scatter(x,y_RBC, label = "RBC") #, c = 'bl', s=100, marker = 'o')
plt.hold(True)
plt.scatter(x,y_ring, label = "Ring")
y40 = y_ring40 - y_RBC40
#plt.axis([1.00, 1.12, 0, 27], fontsize=18)
plt.xlabel('Medium density (g/mL)', fontsize=12)
plt.ylabel('Levitation height (${\mu}m$)', fontsize=12)
plt.legend(loc='lower right', fontsize=11)

plt.figure(3)
plt.figure(figsize=(4.5,4.5))
Gd = 0.01*np.arange(3,14)
x = pow(10,4)*(((-9.05)*pow(10,-6))+(Gd*0.00032*(9.05/0.719)))
y_RBC = h_cell_RBC[:,0]
y_ring = h_cell_ring[:,0]
y = y_ring - y_RBC

max_chi = h_cell_RBC.shape[0]
y_RBCdense = h_cell_RBC[:,max_chi-1]
y_ringdense = h_cell_ring[:,max_chi-1]
ydense = y_ringdense - y_RBCdense
plt.scatter(x,y_RBCdense, label = "RBC") #, c = 'bl', s=100, marker = 'o')
plt.hold(True)
plt.scatter(x,y_ringdense, label = "Ring")
#plt.axis([0.75, 5.5, 0, 27], fontsize=18)
plt.xlabel('Medium magnetic susceptibility (x$10^4$)', fontsize=12)
plt.ylabel('Levitation height (${\mu}m$)', fontsize=12)
plt.legend(loc='upper right', fontsize=11)


# In[ ]:


#Simulation of cell levitation using levitation height outputs from previous equations
RBC30PBS = h_cell_RBC[0,0]
ring30PBS = 369 #369 for mid, h_cell_ring[0,0] by calculation
troph30PBS = h_cell_troph[0,0]
schiz30PBS = h_cell_schiz[0,0]
RBC30Perc = h_cell_RBC[0,max_rho-1]
ring30Perc = h_cell_ring[0,max_rho-1]
troph30Perc = h_cell_troph[0,max_rho-1]
schiz30PBS = h_cell_schiz[0,max_rho-1]

#free parasite density ~ 1.05 g/mL from 1987 paper on P. chabaudi

#Defining cell population size (number of cells) and cell dilution, a cell spread factor based on known density range for cells, 
#percentage parasitemia
RBCsize = 2000 #2500 #size of cell population
RBCscale = 8 #measure of cell spread (how tight the levitation band is)
parasitemia = 0.01
ringsize = RBCsize*parasitemia #size of cell population, for rings, as a percentage of the total RBC population
ringscale = 10 #measure of cell spread (how tight the levitation band is)

#Generate cell height distributions by seeding the above inputs into a random number generator
#Print a plot with this distribution, representing the different cell types as circles of different colors
RBC = np.random.normal(RBC30PBS, RBCscale, RBCsize)
plt.figure(4)
plt.figure(figsize=(6,6))
plt.scatter(np.random.randint(0,1000,RBCsize),RBC, c='r', label = "Uninfected RBC") #,\ mean\ height = 357{\mu}m$")

#Generate ring distribution
ring = np.random.normal(ring30PBS, ringscale, ringsize)
plt.hold(True)
plt.scatter(np.random.randint(0,1000,ringsize),ring,c='g', label = "Ring-stage infected RBC") #,\ mean\ height = 361{\mu}m$")
plt.axis([0, 300, 250, 750])
plt.xlabel('Channel y (${\mu}m$)', fontsize=18)
plt.ylabel('Channel z (${\mu}m$)', fontsize=18)
plt.legend(loc='upper right', fontsize=14)


# In[ ]:


RBC30PBS = h_cell_RBC[0,0]
ring30PBS = 369 #369 for mid, h_cell_ring[0,0] by calculation
troph30PBS = h_cell_troph[0,0]
schiz30PBS = h_cell_schiz[0,0]
agran30PBS = h_cell_agran[0,0]
gran30PBS = h_cell_gran[0,0]
RBC30Perc = h_cell_RBC[0,max_rho-1]
ring30Perc = h_cell_ring[0,max_rho-1]
troph30Perc = h_cell_troph[0,max_rho-1]
schiz30Perc = h_cell_schiz[0,max_rho-1]
agran30Perc = h_cell_agran[0,max_rho-1]
gran30Perc = h_cell_gran[0,max_rho-1]

#free parasite density ~ 1.05 g/mL from 1987 paper on P. chabaudi

axiswidth = 1000 #300
figbottom = 18 #6
RBCsize = 6000 #2000 #size of cell population
RBCscale = 8 #measure of cell spread (how tight the levitation band is)
parasitemia = 0.01
ringsize = RBCsize*parasitemia #size of cell population, for rings, as a percentage of the total RBC population
ringscale = 10 #measure of cell spread (how tight the levitation band is)
agransize = RBCsize*0.35*(2*10**(-3))
gransize = RBCsize*0.65*(2*10**(-3))
WBCscale = 8

#Generate RBC distribution
RBC = np.random.normal(RBC30Perc, RBCscale, RBCsize)
plt.figure(4)
plt.figure(figsize=(figbottom,6))
plt.scatter(np.random.randint(0,1000,RBCsize),RBC, c='r', label = "Uninfected RBC") #,\ mean\ height = 357{\mu}m$")

#Generate ring distribution
ring = np.random.normal(ring30Perc, ringscale, ringsize)
plt.hold(True)
plt.scatter(np.random.randint(0,1000,ringsize),ring,c='g', label = "Ring-stage infected RBC") #,\ mean\ height = 361{\mu}m$")

#Generate WBC distribution
agran = np.random.normal(agran30Perc,WBCscale,agransize)
gran = np.random.normal(gran30Perc,WBCscale,gransize)
plt.hold(True)
plt.scatter(np.random.randint(0,1000,agransize),agran,c='c', label = "Agranulocytes") #,\ mean\ height = 361{\mu}m$")
plt.hold(True)
plt.scatter(np.random.randint(0,1000,gransize),gran,c='b', label = "Granulocytes") #,\ mean\ height = 361{\mu}m$")
plt.axis([0, axiswidth, 250, 750])
plt.xlabel('Channel y (${\mu}m$)', fontsize=18)
plt.ylabel('Channel z (${\mu}m$)', fontsize=18)
plt.legend(loc='upper right', fontsize=14)


# In[ ]:




