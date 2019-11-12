# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22, 2019

@author: VHDR
"""
"""
Spyder Editor

This is a temporary script file.
"""

#from scipy import *
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

    
S = 5 # Define the size of sliding-window (SxS)

### Construction of a synthetic image degraded with haze
im = np.double( imread('cameraman.png') )
Nr,Nc = im.shape 
sigN = 25.0
noise = np.random.normal(0, sigN, [Nr,Nc])
#noise = np.random.laplace(0, sigN, [Nr,Nc])
inp = im + noise

#########################################
# Extend input image for edge processing
A1 = np.concatenate((np.flipud(np.fliplr(inp)), np.flipud(inp), np.flipud(np.fliplr(inp))), axis=1)
A2 = np.concatenate((np.fliplr(inp), inp, np.fliplr(inp)), axis=1)
A3 = np.concatenate((np.flipud(np.fliplr(inp)), np.flipud(inp), np.flipud(np.fliplr(inp))), axis=1)
f_proc = np.concatenate( (A1,A2,A3) ,axis=0)
f_proc = f_proc[Nr-int((S-1)/2):2*Nr+int((S-1)/2), Nc-int((S-1)/2):2*Nc+int((S-1)/2)]

f_mv = np.zeros([S,S]) # Container of the sliding window

out1 = np.zeros([Nr,Nc])
out2 = np.zeros([Nr,Nc])

for k in range(Nr):
    leyend = 'Removing Noise: ' + str(int(100*(k+1)/Nr)) + '%'
    print(leyend)
    for l in range(Nc):
        f_mv = f_proc[ k:S+k, l:S+l ]
        out1[k,l] = np.mean(f_mv)
        out2[k,l] = np.median(f_mv)
print('Done!')
                    
MAE0 = np.mean( abs( im - inp )**2 )
MAE1 = np.mean( abs( im - out1 )**2 )
MAE2 = np.mean( abs( im - out2 )**2 )
leyend = 'Mean Absolute Error: ' + str(MAE1)
print(leyend)

plt.subplot(141)
plt.imshow(im, cmap='gray')
plt.title('Undegraded Image (Reference)')

plt.subplot(142)
plt.imshow((inp), cmap='gray')
plt.title('Noisy Image (Input)')
plt.xlabel('MSE = ' + str(np.float16(MAE0)))

plt.subplot(143)
plt.imshow(out1, cmap='gray')
plt.title('Processed Image (Average Filter)')
plt.xlabel('MSE = ' + str(np.float16(MAE1)))

plt.subplot(144)
plt.imshow(out2, cmap='gray')
plt.title('Processed Image (Median Filter)')
plt.xlabel('MSE = ' + str(np.float16(MAE2)))
