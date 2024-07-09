#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 17:42:10 2018

@author: CW
"""

import numpy as np
from math import sqrt
from scipy.stats import norm
#%matplotlib inline
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')
import matplotlib.pyplot as plt 
import os
import cv2
from matplotlib.lines import Line2D 
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

def dist(x,y,i):
    '''distance bet x and y'''
    return np.sqrt((x[0,i]-y[0,i])**2. + (x[1,i]-y[1,i])**2.)

#parameters
N = 100
dt = 0.5
p0 = 0.1 ; q0 = 0.1 ; s0 = 1.0
p1 = 0.3 ; q1 = 0.2 ; s1 = 0.5
p2 = 1. ; q2 = 1. ; s2 = 0.1
r0 = 70. ; r1 = 10. ; r2 = 1.
lim = 100.

#trajectory variation
N0 = np.random.normal(0,2,size=(2,N))
N1 = np.random.normal(0,2,size=(2,N))
N2 = np.random.normal(0,2,size=(2,N))
#N2 = np.random.randint(2, size=(2, N))-1.

#initial position
m0 = np.random.rand(2,N)*80
m1 = np.random.rand(2,N)*50
m2 = np.random.rand(2,N)*10

#%% 3d normal distibution.. surface potential
#Parameters to set
mu_x = 0
variance_x = 5

mu_y = 0
variance_y = 5

#Create grid and multivariate normal
x = np.round(np.linspace(-1,1,100),2)
y = np.round(np.linspace(-1,1,100),2)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])
z = rv.pdf(pos)

#%%
v0 = np.zeros([2,N])
v1 = np.zeros([2,N])
v2 = np.zeros([2,N])
d01 = np.zeros([1,N]) ; d01[0,0]=dist(m0,m1,0)
d12 = np.zeros([1,N]) ; d12[0,0]=dist(m1,m2,0)
d20 = np.zeros([1,N]) ; d20[0,0] = dist(m0,m2,0)
af10 = np.zeros([2,N]); af20 = np.zeros([2,N]) 
af01 = np.zeros([2,N]); af21 = np.zeros([2,N])
af02 = np.zeros([2,N]); af12 = np.zeros([2,N])
rf10 = np.zeros([2,N]); rf20 = np.zeros([2,N]) 
rf01 = np.zeros([2,N]); rf21 = np.zeros([2,N])
rf02 = np.zeros([2,N]); rf12 = np.zeros([2,N])
m00 = np.zeros([2,N]) ; m11 = np.zeros([2,N]) ; m22 = np.zeros([2,N])
v0[:,1] = (m0[:,1]-m0[:,0])/dt ; #v0[:,1] = (m0[:,2]-m0[:,1])/dt
v1[:,1] = (m1[:,1]-m1[:,0])/dt ; #v1[:,1] = (m1[:,2]-m1[:,1])/dt
v2[:,1] = (m2[:,1]-m2[:,0])/dt ; #v2[:,1] = (m2[:,2]-m2[:,1])/dt
for i in range(1,N-1):
    d01[0,i] = dist(m0,m1,i) 
    d12[0,i] = dist(m1,m2,i)
    d20[0,i] = dist(m0,m2,i)
    
    #affractive force
    af10[:,i] = (1./d01[0,i]**13)*((m1[:,i]-m0[:,i])/d01[0,i])
    af20[:,i] = (1./d20[0,i]**13)*((m2[:,i]-m0[:,i])/d20[0,i])
    
    #repulasive force
    rf10[:,i] = (1./d01[0,i]**7)*((m1[:,i]-m0[:,i])/d01[0,i])
    rf20[:,i] = (1./d20[0,i]**7)*((m2[:,i]-m0[:,i])/d20[0,i])
    
    #main loop
    v0[:,i+1] = v0[:,i]+(p0*(m0[:,0]-m0[:,i])\
      +q0*(np.heaviside(r0-d01[0,i],0)*af10[:,i]\
      +np.heaviside(r0-d20[0,i],0)*af20[:,i])\
      -s0*(np.heaviside(r2-d01[0,i],0)*rf10[:,i]\
           +np.heaviside(r2-d20[0,i],0)*rf20[:,i]))*dt
    
    af01[:,i] = (1./d01[0,i]**13)*((m0[:,i]-m1[:,i])/d01[0,i])
    af21[:,i] = (1./d12[0,i]**13)*((m2[:,i]-m1[:,i])/d12[0,i])
    rf01[:,i] = (1./d01[0,i]**7)*((m0[:,i]-m1[:,i])/d01[0,i])
    rf21[:,i] = (1./d12[0,i]**7)*((m2[:,i]-m1[:,i])/d12[0,i])
    v1[:,i+1] = v1[:,i]+(p1*(m1[:,0]-m1[:,i])\
      +q1*(np.heaviside(r1-d01[0,i],0)*af01[:,i]\
      +np.heaviside(r1-d12[0,i],0)*af21[:,i])\
      -s1*(np.heaviside(r1-d01[0,i],0)*rf01[:,i]\
           +np.heaviside(r1-d12[0,i],0)*rf21[:,i]))*dt
    
    af02[:,i] = (1./d20[0,i]**13)*((m0[:,i]-m2[:,i])/d20[0,i])
    af12[:,i] = (1./d12[0,i]**13)*((m1[:,i]-m2[:,i])/d12[0,i])
    rf02[:,i] = (1./d20[0,i]**7)*((m0[:,i]-m2[:,i])/d20[0,i])
    rf12[:,i] = (1./d12[0,i]**7)*((m1[:,i]-m2[:,i])/d12[0,i])
    v2[:,i+1] = v2[:,i]+(p2*(m2[:,0]-m2[:,i])\
      +q2*(np.heaviside(r2-d20[0,i],0)*af02[:,i]\
      +np.heaviside(r2-d12[0,i],0)*af12[:,i])\
      -s2*(np.heaviside(r0-d20[0,i],0)*rf02[:,i]\
           +np.heaviside(r0-d12[0,i],0)*rf12[:,i]))*dt
    
    #boundary limit
    m00[:,i+1] = dt*(v0[:,i+1]-v0[:,i])+2.*m0[:,i]-m0[:,i-1]+N0[:,i]
    m11[:,i+1] = dt*(v1[:,i+1]-v1[:,i])+2.*m1[:,i]-m1[:,i-1]+N1[:,i]
    m22[:,i+1] = dt*(v2[:,i+1]-v2[:,i])+2.*m2[:,i]-m2[:,i-1]+N2[:,i]
    for k in range(2):
        if m00[k,i+1] <= lim and m00[k,i+1] >= 0.: m0[k,i+1] = m00[k,i+1]
        elif m00[k,i+1] > lim: m0[k,i+1] = 2*lim-m00[k,i+1] 
        elif m00[k,i+1] < 0.: m0[k,i+1] = -m00[k,i+1]
    for k in range(2):
        if m11[k,i+1] <= lim and m11[k,i+1] >= 0.: m1[k,i+1] = m11[k,i+1]
        elif m11[k,i+1] > lim: m1[k,i+1] = 2*lim-m11[k,i+1] 
        elif m11[k,i+1] < 0.: m1[k,i+1] = -m11[k,i+1]
    for k in range(2):
        if m22[k,i+1] <= lim and m22[k,i+1] >= 0.: m2[k,i+1] = m22[k,i+1]
        elif m22[k,i+1] > lim: m2[k,i+1] = 2*lim-m22[k,i+1] 
        elif m22[k,i+1] < 0.: m2[k,i+1] = -m22[k,i+1]

#%%    
F = plt.figure(1) 
f = F.add_subplot(111)
plt.xlabel('x')
plt.ylabel('y')
f.grid(True)
f.plot(m0[0,:],m0[1,:],'.-',alpha=0.2,c='r',label='dominant')
f.plot(m0[0,0],m0[1,0],'ro')
f.plot(m0[0,-1],m0[1,-1],'ko')

f.plot(m1[0,:],m1[1,:],'.-',alpha=0.2,c='g',label='sub0')
f.plot(m1[0,0],m1[1,0],'go')
f.plot(m1[0,-1],m1[1,-1],'ks')

f.plot(m2[0,:],m2[1,:],'.-',alpha=0.2,c='b',label='sub1')
f.plot(m2[0,0],m2[1,0],'bo')
f.plot(m2[0,-1],m2[1,-1],'kx')

xlim = plt.gca().get_xlim()
ylim = plt.gca().get_ylim()
handles, labels = f.get_legend_handles_labels()
plt.legend(handles, labels, bbox_to_anchor=(0.8,-0.2), loc="center left",prop={'size':7},borderaxespad=0.)
plt.tight_layout(pad=4)

#%%
path1 = "/Users/CW/Modelling/"
os.chdir(path1)    
folder = 'GBD1.3'  
os.mkdir(folder)
#save the plot
F = plt.figure(2) 
f = F.add_subplot(111)
plt.xlabel('x')
plt.ylabel('y')
f.grid(True)
f.set_xlim(xlim)
f.set_ylim(ylim)
for i in range(N-1):
    f.scatter(m0[0,i],m0[1,i],marker='o',alpha=0.2,c='r',s=10,label='dominant')
    l = Line2D([m0[0,i],m0[0,i+1]],[m0[1,i],m0[1,i+1]],c='r',linewidth=0.5)                                    
    f.add_line(l)
    f.plot(m0[0,0],m0[1,0],'ro')
    f.plot(m0[0,-1],m0[1,-1],'ko')
    
    f.scatter(m1[0,i],m1[1,i],marker='o',alpha=0.2,c='g',s=10,label='sub0')
    l = Line2D([m1[0,i],m1[0,i+1]],[m1[1,i],m1[1,i+1]],c='g',linewidth=0.5)                                    
    f.add_line(l)
    f.plot(m1[0,0],m1[1,0],'go')
    f.plot(m1[0,-1],m1[1,-1],'ks')
    
    f.scatter(m2[0,i],m2[1,i],marker='o',alpha=0.2,c='b',s=10,label='sub1')
    l = Line2D([m2[0,i],m2[0,i+1]],[m2[1,i],m2[1,i+1]],c='b',linewidth=0.5)                                    
    f.add_line(l)
    f.plot(m2[0,0],m2[1,0],'bo')
    f.plot(m2[0,-1],m2[1,-1],'kx')
    plt.savefig(os.path.join(folder,'trj'+str(i)+'.jpg'))
    
#%% make a movie
path2 = "/Users/CW/Modelling/"+folder
os.chdir(path2)    
im=[]
for i in range(N-1):
    im.append(cv2.imread('trj'+str(i)+'.jpg'))

height,width,layers=im[1].shape
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
video=cv2.VideoWriter('GBD1.1.avi',fourcc,2,(width,height))

for j in range(N-1):
    video.write(im[j])
cv2.destroyAllWindows()
video.release()   
