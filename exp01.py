# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22, 2019

@author: VHDR
"""
"""
Spyder Editor

This is a temporary script file.
"""

# Import required libraries
from scipy import *
import numpy as np
import matplotlib.pyplot as plt


def rgb2gray(img):
    img_gray = double( (0.2989*img[:,:,0] + 0.5870*img[:,:,0] + 0.1140*img[:,:,0])) # RGB to Gray Conversion
    return img_gray
        
# Read image from disk
im = 255*double( imread("peppers.png") )
# Obtain image size
Nr,Nc = im.shape

Nsample = 100 #Number of noisy samples

ens = np.zeros([Nr,Nc,Nsample]) #Contained for nosy image stochestic process

sigN = 25.0 #Additive noise variance

#Create a nosy image ensamble
for k in range(Nsample):
    noise = sigN * np.random.randn(Nr,Nc)
    ens[:,:,k] = im + noise
    plt.figure(1)
    plt.title('Stochastic process of the noisy image')
    plt.imshow(abs(ens[:,:,k]), cmap='gray')
    pause(0.001)
    plt.show()
    plt.clf()
    
out = zeros([Nr,Nc]) #Container of the processed image

#Estimation (Maximum-likelihood estimator) of the output image
for k in range(Nr):
    for l in range(Nc):
        out[k,l] = mean(ens[k,l,:])
        
inp  = ens[:,:,0]
print('Done!')

                        
MAE = mean( abs( im - out )**2 ) #Calculation of the Mean-Absolute-Error

leyend = 'Mean Absolute Error: ' + str(MAE)
print(leyend)
#y2 = y2/y2[:].max()
subplot(131), plt.imshow(uint8(im), cmap='gray'), plt.title('Undegraded Image')
subplot(132), plt.imshow(uint8(abs(inp)), cmap='gray'), plt.title('Noisy Image')
subplot(133), plt.imshow(uint8(abs(out)), cmap='gray'), plt.title('Processed Image')
