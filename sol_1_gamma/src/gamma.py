'''
Created on Mar 9, 2017

@author: vlad
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma_correction(img, correction):
    result = img / 255.0
    result = cv2.pow(result, correction)
    return np.uint8(result * 255)


def log_correction(img):
    result   = img / 255.0
    result = np.ones(result.shape) + result
    result = cv2.log(result)
    return np.uint8(result * 255)

def neg_correction(img):
    result = img / 255.0
    result = np.ones(result.shape) - result
    return np.uint8(result * 255)

alpha1 = 0.8
alpha2 = 1.8
img = cv2.imread('test1.jpg', cv2.IMREAD_GRAYSCALE)
gamma1_img = gamma_correction(img, alpha1)
gamma2_img = gamma_correction(img, alpha2)
log_img = log_correction(img)
neg_img = neg_correction(img)

# cv2.imshow("result {}".format(alpha), gamma2_img)
# cv2.waitKey()
# cv2.imshow("log result", log_img)
# cv2.waitKey()


fig = plt.figure()

a = fig.add_subplot(2,3,1)
a.set_title('Original img')
imgplot = plt.imshow(img, cmap='gray')

a = fig.add_subplot(2,3,2)
a.set_title('Log img')
imgplot = plt.imshow(log_img, cmap='gray')

a = fig.add_subplot(2,3,3)
a.set_title('Gamma {} img'.format(alpha1))
imgplot = plt.imshow(gamma1_img, cmap='gray')

a=fig.add_subplot(2,3,4)
a.set_title('Gamma {} img'.format(alpha2))
imgplot = plt.imshow(gamma2_img, cmap='gray')

a=fig.add_subplot(2,3,5)
a.set_title('Negative img')
imgplot = plt.imshow(neg_img, cmap='gray')


plt.show()
print "Program is over"