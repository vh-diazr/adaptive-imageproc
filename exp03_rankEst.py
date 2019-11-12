import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from pylab import pause
from skimage import io
from skimage.color import rgb2gray
from skimage import data 
from scipy import stats

#def mean_est(sample):
#    estimate = np.sum(sample)/np.size(sample)
#    return estimate
#
#
#def median_est(sample):
#    var_row = np.sort(np.ravel(sample))
#    estimate = var_row[int((np.size(var_row)-1.)/2.)]
#    return estimate

def mse(reference,input):
    error = np.sum((reference-input)**2)/np.size(input)
    return error


def alpha_trimmed_mean(sample):
   # sort_array = np.sort(np.ravel(sample))
    proc = stats.trim_mean(np.ravel(sample), 0.4)
    #proc = np.sum(trimmed_mean)/np.size(trimmed_mean)
    return proc


#def rankEV_est(sample, ev):    
#    central_pixel = sample[int((np.size(sample,0)-1)/2),int((np.size(sample,1)-1)/2)]
#    res_vector = np.ravel( np.zeros(1))
#    in_vector = np.ravel(sample)    
#    for k in range (np.size(in_vector)):
#        if (central_pixel - ev) <= in_vector[k] and in_vector[k] <= (central_pixel + ev):
#            res_vector = np.append(res_vector,in_vector[k])
#    res_vector = np.delete(res_vector,[0])
#    if np.size(res_vector) <= 5:
#        prom = np.median(sample)
#    else:
#        prom = np.sum(res_vector)/np.size(res_vector)          
#    return prom      

def rankKNB_est(mw,K):
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
    prom = np.sum(nbh)/np.size(nbh)
    return prom

def rankEV_est(sample, ev):    
    central_pixel = sample[int((np.size(sample,0)-1)/2),int((np.size(sample,1)-1)/2)]
    xr,yr = np.where( (sample >= (central_pixel - ev)) & (sample <= (central_pixel + ev)) )
    res_vector = np.zeros( np.size(xr) )
    for k in range(np.size(res_vector)):
        res_vector[k] = sample[xr[k],yr[k]]
    if np.size(res_vector) <= 5:
        prom = np.median(sample)
    else:
        prom = np.sum(res_vector)/np.size(res_vector)          
    return prom      
         
#def rankEV_est(mw,ev):
#    r,c = mw.shape
#    cent = mw[int(floor(r/2)), int(floor(c/2))]
#    x,y = where( mw >= (cent-ev) )
#    nbh0 = ravel(mw[x,y])
#    x = 0
#    x = where( nbh0 <= (cent + ev) )
#    nbh = nbh0[x]
#    if np.size(nbh) <= 5:
#        prom = np.median(nbh)
#    else:
#        prom = np.sum(nbh)/np.size(nbh)          
#    return prom      

s = rgb2gray(data.camera())
Nr,Nc = np.shape(s)
var_n = 0.05 * 255
#n  = np.random.normal(0,var_n,[Nr,Nc]) #se genera el ruido estacionario con distribucion Normal
n = np.random.laplace(0,var_n,[Nr,Nc]) #se genera el ruido estacionario con distribucion de Laplace  

f = np.abs(s + n)
f1 = np.concatenate((np.flipud(np.fliplr(f)), np.flipud(f), np.flipud(np.fliplr(f))), axis = 1)
f2 = np.concatenate((np.fliplr(f),f, np.fliplr(f)), axis = 1)
f3 = np.concatenate((np.flipud(np.fliplr(f)),np.flipud(f),np.flipud(np.fliplr(f))), axis = 1)
f_proc = np.concatenate((f1,f2,f3), axis = 0)

s_est1 = np.zeros([Nr,Nc])
s_est2 = np.zeros([Nr,Nc])

Nw = 11
counter = 0

for k in range (Nr):
   for l in range(Nc):
       window = f_proc[Nr-int((Nw - 1)/2)+k:Nr-int((Nw-1)/2)+k+Nw,Nc-int((Nw-1)/2)+l:Nc-int((Nw-1)/2)+l+Nw]
       leyend = 'Removing Noise: ' + str(int(100*(k+1)/Nr)) + '%'
       print(leyend)
       s_est1[k,l] = rankEV_est(window,45)
       s_est2[k,l] = rankKNB_est(window,65)

mse1 = mse(s,s_est1)
mse2 = mse(s,s_est2)

plt.subplot(221), plt.imshow(f, cmap='gray'), plt.title('imagen degradada')
plt.subplot(222), plt.imshow(s, cmap='gray'), plt.title('image no degradada, mse = ' + str(0))
plt.subplot(223), plt.imshow(s_est2, cmap='gray'), plt.title('filtro rankKNB_est, mse = ' + str(mse2))
plt.subplot(224), plt.imshow(s_est1, cmap='gray'), plt.title('filtro rankEV_est, mse = ' + str(mse1))
plt.show()










##plt.imshow(s_estimada, cmap = 'gray')
