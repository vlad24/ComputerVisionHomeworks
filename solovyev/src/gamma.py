'''
Created on Mar 9, 2017

@author: vlad
'''

import cv2

import matplotlib.pyplot as plt
import numpy as np


def gamma_correction(img, correction):
    result = img[:]
    result = result / 255.0
    result = cv2.pow(result, correction)
    return np.uint8(result * 255)


def log_correction(img):
    result   = np.copy(img)
    result   = result / 255.0
    result = np.ones(result.shape) + result
    result = cv2.log(result)
    return np.uint8(result * 255)

def neg_correction(img):
    result = np.copy(img)
    result = result / 255.0
    result = np.ones(result.shape) - result
    return np.uint8(result * 255)

def p_linear_correction(img, t1, t2, k1, k2):
    result = np.zeros(img.shape)
    for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j]/255.0 < t1:
                    result[i][j] = img[i][j] * k1 / 255.0
                if img[i][j]/255.0 > t2:
                    result[i][j] = img[i][j]* k2 / 255.0
    return np.uint8(result * 255)

def linear_correction(img, k):
    result = np.copy(img)
    result = result / 255.0
    result *= k 
    return np.uint8(result * 255)

img = cv2.imread('../img/V/1/original.jpg', cv2.IMREAD_GRAYSCALE)
alpha1 = 0.6
alpha2 = 1.8
gamma1_img = gamma_correction(img, alpha1)
gamma2_img = gamma_correction(img, alpha2)
log_img = log_correction(img)
neg_img = neg_correction(img)
lin_img = linear_correction(img, 1.5)
plin_img = p_linear_correction(img, 0.2, 0.3, 0.1, 1.5)

cv2.imwrite("small_alpha_image.jpeg", gamma1_img)
cv2.imwrite("big_alpha_image.jpeg", gamma2_img)
cv2.imwrite("log_image.jpeg", log_img)
cv2.imwrite("negative_image.jpeg", neg_img)
cv2.imwrite("lin_image.jpeg", lin_img)
cv2.imwrite("plin_image.jpeg", plin_img)
cv2.imwrite("orig-log_image.jpeg", img-log_img)




print "Program is over"