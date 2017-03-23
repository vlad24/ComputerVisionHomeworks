'''
Created on Mar 23, 2017

@author: vlad
'''

import numpy as np
import cv2


def get_balanced_avg_mask(n):
    return np.ones((n, n),np.float32)/ (n**2)

def get_weighted_mask(n):
    m = np.zeros((n,n))
    c = n/2
    for i in range(-n/2, n/2):
        a = 1
        m[c + i][c]     = a
        m[c]    [c + i] = a
    s = np.sum(m)
    return m / s

     
n = 11
image = cv2.imread('../img/V/3/test3.png', cv2.IMREAD_GRAYSCALE)
m1 = get_balanced_avg_mask(n)
avg_img  = cv2.filter2D(image, -1, m1)
m2 = get_weighted_mask(n)
print m2
special_img = cv2.filter2D(image, -1, m2)
median_img = cv2.medianBlur(image, n)


cv2.imwrite("../img/V/3/balanced.jpg",  avg_img)
cv2.imwrite("../img/V/3/weighted.jpg",  special_img)
cv2.imwrite("../img/V/3/medianed.jpg", median_img)
print "Done"