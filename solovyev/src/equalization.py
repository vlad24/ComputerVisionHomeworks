'''
Created on Mar 16, 2017

@author: vlad
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

def equalization(img):
    hist = np.zeros(255)
    result = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            T = img[i][j]
            hist[T] += 1
    hist = hist / (img.shape[0] * img.shape[1])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            T = img[i][j]
            result[i][j] = sum(hist[0:T])
    return np.uint8(result * 255)

def local_equalization(img, P, L):
    H = img.shape[0]
    W = img.shape[1]
    result = np.zeros(img.shape)
    A = H * W
    a = 0
    for i in range(H):
        for j in range(W):
            local_hist = np.zeros(255)
            for k in range(i - L/2, i + L/2 + 1):
                for l in range(j - P/2, j + P/2 + 1):
                    if (0 <= k < H and 0 <= l < W):
                        T = img[k][l]
                        local_hist[T] += 1
            local_hist = local_hist / (P * L)
            T = img[i][j]
            result[i][j] = sum(local_hist[0:T])
            a += 1
            print "Done: {} \r".format(float(a)/A * 100)
    return np.uint8(result * 255)

def hist(img):
    plt.hist(img.ravel(),256,[0,256])
    plt.show()

    
image = cv2.imread('../img/V/2/slon.jpg', cv2.IMREAD_GRAYSCALE)
hist(image)
equalization_image = equalization(image)
hist(equalization_image)
lequalization_image = local_equalization(image, 71, 71)
hist(lequalization_image)
cv2.imwrite("../img/V/slon_LEqualization.jpg", lequalization_image)
print "Done"

