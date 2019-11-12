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


def MSE_est(sample, sigma_n):
    central = sample[int((np.size(sample,0)-1)/2),int((np.size(sample,1)-1)/2)]
    mu = np.mean(sample)
    
    sigma_sample = np.var(sample)
    sigma_s = np.max([0.01,(sigma_sample-sigma_n)])
    G = sigma_s / (sigma_s + sigma_n)
    prom = mu + G*(central - mu)
    return prom      
         

s = rgb2gray(data.camera())
Nr,Nc = np.shape(s)
var_n = 0.05 * 255
#n  = np.random.normal(0,var_n,[Nr,Nc]) #se genera el ruido estacionario con distribucion Normal
n = np.random.laplace(0,var_n,[Nr,Nc]) #se genera el ruido estacionario con distribucion de Laplace  
sigma_n = np.var(n)
f = np.real(s + n)
f1 = np.concatenate((np.flipud(np.fliplr(f)), np.flipud(f), np.flipud(np.fliplr(f))), axis = 1)
f2 = np.concatenate((np.fliplr(f),f, np.fliplr(f)), axis = 1)
f3 = np.concatenate((np.flipud(np.fliplr(f)),np.flipud(f),np.flipud(np.fliplr(f))), axis = 1)
f_proc = np.concatenate((f1,f2,f3), axis = 0)

s_est1 = np.zeros([Nr,Nc])
s_est2 = np.zeros([Nr,Nc])

Nw = 9
counter = 0

for k in range (Nr):
   for l in range(Nc):
       window = f_proc[Nr-int((Nw - 1)/2)+k:Nr-int((Nw-1)/2)+k+Nw,Nc-int((Nw-1)/2)+l:Nc-int((Nw-1)/2)+l+Nw]
       leyend = 'Removing Noise: ' + str(int(100*(k+1)/Nr)) + '%'
       print(leyend)
       s_est1[k,l] = MSE_est(window,sigma_n)
       #s_est2[k,l] = rankKNB_est(window,65)

mse1 = mse(s,s_est1)
#mse2 = mse(s,s_est2)

plt.subplot(131), plt.imshow(f, cmap='gray'), plt.title('imagen degradada')
plt.subplot(132), plt.imshow(s, cmap='gray'), plt.title('image no degradada, mse = ' + str(0))
plt.subplot(133), plt.imshow(np.abs(s_est1), cmap='gray'), plt.title('filtro MSE_est, mse = ' + str(mse1))
#plt.subplot(224), plt.imshow(s_est1, cmap='gray'), plt.title('filtro rankEV_est, mse = ' + str(mse1))
plt.show()










##plt.imshow(s_estimada, cmap = 'gray')
