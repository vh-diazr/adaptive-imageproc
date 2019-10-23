# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 09:08:06 2016

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 12:11:17 2016

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from scipy import *
from pylab import *
from scipy.misc import imresize
from scipy.misc import imsave
import cv2
from cv2.ximgproc import guidedFilter
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral

def KNB(mw,K):
    r,c = mw.shape
    cent = mw[int(floor(r/2)), int(floor(c/2))]
    dist = zeros([r,c])
    dist = abs(mw - cent)
    dist_ord = sort(ravel(dist[:]))
    dist_K = dist_ord[K-1]
    x,y = where(dist <= dist_K)
    nbh = mw[x,y]
    nbh = ravel(sort(nbh))
    nbh = nbh[0:K]
    return nbh

def EV(mw,ev):
    r,c = mw.shape
    cent = mw[int(floor(r/2)), int(floor(c/2))]
    x,y = where( mw >= (cent-ev) )
    nbh0 = ravel(mw[x,y])
    x = where( nbh0 <= (cent + ev) )
    nbh = nbh0[x]
    return nbh

def rgb2gray(img):
    img_gray = double( (0.2989*img[:,:,0] + 0.5870*img[:,:,0] + 0.1140*img[:,:,0])) # RGB to Gray Conversion
    return img_gray
        
### Construction of a synthetic image degraded with haze
im = 255*double( imread("peppers.png") )
Nr,Nc = im.shape

Nsample = 9

ens = zeros([Nr,Nc,Nsample])
sigN = 25.0
for k in range(Nsample):
    noise = sigN * np.random.randn(Nr,Nc)
    ens[:,:,k] = im + noise
    figure(1), imshow(abs(ens[:,:,k])), gray()
    pause(0.001)
    
out = zeros([Nr,Nc])
for k in range(Nr):
    for l in range(Nc):
        out[k,l] = mean(ens[k,l,:])
        
inp  = ens[:,:,0]
print('Done!')

                        
MAE = mean( abs( im - out )**2 )

leyend = 'Mean Absolute Error: ' + str(MAE)
print(leyend)
#y2 = y2/y2[:].max()
subplot(131), imshow(uint8(im)), title('Undegraded Image'), gray()
subplot(132), imshow(uint8(abs(inp))), title('Noisy Image')
subplot(133), imshow(uint8(abs(out))), title('Processed Image')
