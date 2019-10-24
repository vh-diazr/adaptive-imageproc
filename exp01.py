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
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from pylab import pause


def rgb2gray(img):
    img_gray = double( (0.2989*img[:,:,0] + 0.5870*img[:,:,0] + 0.1140*img[:,:,0])) # RGB to Gray Conversion
    return img_gray
        
# Read image from disk
im = np.double( imread("peppers.png") )
# Obtain image size
Nr,Nc = im.shape

Nsample = 50 #Number of noisy samples

#ens = np.zeros([Nr,Nc,Nsample]) #Contained for nosy image stochestic process
inp = np.zeros([Nr,Nc])

sigN = 25.0 #Additive noise variance
out = np.zeros([Nr,Nc]) #Container of the processed image
acc = np.zeros([Nr,Nc])
MSE = np.zeros(Nsample)
#Create a nosy image ensamble
for q in range(Nsample):
    noise = sigN * np.random.randn(Nr,Nc)
    inp = im + noise
    plt.figure(1)
    plt.title('Stochastic process of the noisy image')
    plt.imshow(abs(inp), cmap='gray')
    pause(0.01)
    plt.show()
    plt.clf()
    
    # Maximum-likelihood estimion of the expected value of the output image (FAST)
    acc = acc + inp
    out = acc / (q+1)
    
    #out = np.mean(ens,2) # Maximum-likelihood estimion of the output image (FAST)
    
    # Maximum-likelihood estimion of the output image (SLOW)
    #for k in range(Nr):
    #    for l in range(Nc):
    #        out[k,l] = np.mean(ens[k,l,np.arange(q)])
        
    MSE[q] = np.mean( abs( im - out )**2 ) #Calculation of the Mean-Absolute-Error
print('Done!')
plt.close()

                        
leyend = 'Mean Squared Error: ' + str(MSE[q])
print(leyend)
plt.figure(1)
plt.subplot(131), plt.imshow(im, cmap='gray'), plt.title('Undegraded Image (Reference)')
plt.subplot(132), plt.imshow(abs(inp), cmap='gray'), plt.title('Noisy Image (Input)')
plt.subplot(133), plt.imshow(abs(out), cmap='gray'), plt.title('Processed Image (Output)')

plt.figure(2)
plt.plot(MSE,'-.*'), plt.grid(), plt.xlabel('No. of noisy images (sample size)'), plt.ylabel('Mean Squared Error (MSE)'), plt.title('Performance of the MLE estimator')
