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
from skimage.io import imread
#from scipy.misc import imresize
#from scipy.misc import imsave
#import cv2
#from cv2.ximgproc import guidedFilter
#from skimage.restoration import denoise_tv_chambolle, denoise_bilateral

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
    x = 0
    x = where( nbh0 <= (cent + ev) )
    nbh = nbh0[x]
    return nbh

def rgb2gray(img):
    img_gray = double( (0.2989*img[:,:,0] + 0.5870*img[:,:,0] + 0.1140*img[:,:,0])) # RGB to Gray Conversion
    return img_gray
        
    
S = 15 # Define the size of sliding-window (SxS)
#f_c = imread("hazy_boats.png") # Read Input Hazy Image
#f_sin = 255*double(imread("original.png")) # Read Original Undegraded Image

### Construction of a synthetic image degraded with haze
im = double( imread("cameraman.png") )
Nr,Nc = im.shape 

sigN = 25.0
noise = np.random.normal(0, sigN, [Nr,Nc])

#sigN = 20.0
#noise = sigN * np.random.randn(Nr,Nc)
inp = im + noise

#########################################
# Extend input image for edge processing
A1 = np.concatenate((np.flipud(np.fliplr(inp)), np.flipud(inp), np.flipud(np.fliplr(inp))), axis=1)
A2 = np.concatenate((np.fliplr(inp), inp, np.fliplr(inp)), axis=1)
A3 = np.concatenate((np.flipud(np.fliplr(inp)), np.flipud(inp), np.flipud(np.fliplr(inp))), axis=1)
f_proc = np.concatenate( (A1,A2,A3) ,axis=0)
f_proc = f_proc[Nr-int((S-1)/2):2*Nr+int((S-1)/2), Nc-int((S-1)/2):2*Nc+int((S-1)/2)]

#pause
#A_test = zeros([Nr,Nc])
#A_test2 = zeros([Nr,Nc])
f_mv = zeros([S,S])
#K = floor(2*S*S/3)
out = zeros([Nr,Nc])
for k in range(Nr):
    leyend = 'Removing Noise: ' + str(int(100*(k+1)/Nr)) + '%'
    print(leyend)
    for l in range(Nc):
        f_mv = f_proc[ k:S+k, l:S+l ]
        f_amv = EV(f_mv, 40)
        out[k,l] = mean(f_amv)
        #figure(1), imshow(f_mv, interpolation='None'), gray(), pause(1e-3), close() 
print('Done!')
                    
MAE = mean( abs( im - out )**2 )
leyend = 'Mean Absolute Error: ' + str(MAE)
print(leyend)

subplot(131), imshow(im), title('Undegraded Image'), gray()
subplot(132), imshow(inp), title('Noisy Image')
subplot(133), imshow(out), title('Undegraded Image')
