import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
from scipy import signal
import time

img=mpimg.imread('houses.jpg')
plt.imshow(img, cmap="gray")

dx = np.zeros( (3,3) )
dx[1,0] = 0.5
dx[1,2] = -0.5
dy = np.zeros( (3,3) )
dy[0,1] = 0.5
dy[2,1] = -0.5
mpimg.imsave('dx.jpg',dx,cmap="gray")
mpimg.imsave('dy.jpg',dy,cmap="gray")
plt.imshow(dx, cmap="gray")
plt.imshow(dy, cmap="gray")

dx_img = signal.convolve2d(img, dx, boundary='symm', mode='same')
dy_img = signal.convolve2d(img, dy, boundary='symm', mode='same')
dm_img = np.sqrt(np.square(dx_img)+np.square(dy_img))
mpimg.imsave('dx_houses.jpg',dx_img,cmap="gray")
mpimg.imsave('dy_houses.jpg',dy_img,cmap="gray")
mpimg.imsave('dm_houses.jpg',dm_img,cmap="gray")
plt.imshow(dx_img, cmap="gray")
plt.imshow(dy_img, cmap="gray")
plt.imshow(dm_img, cmap="gray")

sdx = np.zeros( (3,3) )
sdx[0,0] = 0.125
sdx[0,2] = -0.125
sdx[1,0] = 0.25
sdx[1,2] = -0.25
sdx[2,0] = 0.125
sdx[2,2] = -0.125
sdy = np.zeros( (3,3) )
sdy[0,0] = 0.125
sdy[2,0] = -0.125
sdy[0,1] = 0.25
sdy[2,1] = -0.25
sdy[0,2] = 0.125
sdy[2,2] = -0.125
mpimg.imsave('sdx.jpg',sdx,cmap="gray")
mpimg.imsave('sdy.jpg',sdy,cmap="gray")
plt.imshow(sdx, cmap="gray")
plt.imshow(sdy, cmap="gray")

sdx_img = signal.convolve2d(img, sdx, boundary='symm', mode='same')
sdy_img = signal.convolve2d(img, sdy, boundary='symm', mode='same')
sdm_img = np.sqrt(np.square(sdx_img)+np.square(sdy_img))
mpimg.imsave('sdx_houses.jpg',sdx_img,cmap="gray")
mpimg.imsave('sdy_houses.jpg',sdy_img,cmap="gray")
mpimg.imsave('sdm_houses.jpg',sdm_img,cmap="gray")
plt.imshow(sdx_img, cmap="gray")
plt.imshow(sdy_img, cmap="gray")
plt.imshow(sdm_img, cmap="gray")
