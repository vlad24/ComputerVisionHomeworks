'''
Created on Apr 14, 2017

@author: vlad
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import dtype

k = 5

test_img1_path = "../img/V/{task}/test{task}.jpg"
new_img_path  = "../img/V/{task}/_{name}.jpg"


def fourier_transform(img):
    return np.fft.fftshift(np.fft.fft2(img))

def fourier_transform_inverse(fourier_transform):
    return np.abs(np.fft.ifft2(np.fft.ifftshift(fourier_transform)))

def zerofy_rectangle(f, y_offset, x_offset):
    r = np.copy(f)
    row_c, col_c = (r.shape[0] / 2, r.shape[1] / 2) 
    if (x_offset == 0 and y_offset == 0):
        r[row_c, col_c] = 0
    else:
        r[row_c - y_offset : row_c + y_offset + 1, col_c - x_offset : col_c + x_offset + 1] = 0
    return r

def zerofy_inv(f, y_offset, x_offset):
    r = np.zeros_like(f)
    row_c, col_c = (r.shape[0] / 2, r.shape[1] / 2) 
    r[row_c - y_offset : row_c + y_offset + 1, col_c - x_offset : col_c + x_offset + 1] = f[row_c - y_offset : row_c + y_offset + 1, col_c - x_offset : col_c + x_offset + 1]
    return r
    

def to_magnitude_spectrum(t):
    return np.log(1+np.abs(t))

def plot_part(part, name, img):
    plt.subplot(part)
    plt.title(name)
    plt.imshow(img, cmap = 'gray')
    plt.xticks([]); plt.yticks([])


if __name__ == '__main__':
    r = 25
    img = cv2.imread(test_img1_path.format(task=k), cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(test_img1_path.format(task=k-1), cv2.IMREAD_GRAYSCALE)
    ft1 = fourier_transform(img)
    ft3 = fourier_transform(img3)
    hp1 = zerofy_rectangle(ft1, r, r)
    hp3 = zerofy_inv(ft3, r/2, r/2)
    pr1 = zerofy_rectangle(ft1, 0, 0)
    pr3 = zerofy_rectangle(ft3, 0, 0)
    
     
    plot_part(221, "Original",             img)
    plot_part(222, "Fourier Transform",    to_magnitude_spectrum(ft1))
    plot_part(223, "High Pass Image",      fourier_transform_inverse(hp1))
    plot_part(224, "HP Fourier Transform", to_magnitude_spectrum(hp1))
    plt.show()
      
    ft1 = fourier_transform(img3)
    hp = zerofy_inv(ft1, r/2, r/2)
    plot_part(221, "Original",             img3)
    plot_part(222, "Fourier Transform",    to_magnitude_spectrum(ft3))
    plot_part(223, "Result Image",      fourier_transform_inverse(hp3))
    plot_part(224, "Res Fourier Transform", to_magnitude_spectrum(hp3))
    plt.show()
     
    plot_part(221, "Original",             img)
    plot_part(222, "Fourier Transform",    to_magnitude_spectrum(ft1))
    plot_part(223, "Pr Image",             fourier_transform_inverse(pr1))
    plot_part(224, "Res Fourier Transform", to_magnitude_spectrum(pr1))
    plt.show()
      
    plot_part(221, "Original",             img3)
    plot_part(222, "Fourier Transform",    to_magnitude_spectrum(ft3))
    plot_part(223, "Result Image",         fourier_transform_inverse(pr3))
    plot_part(224, "Res Fourier Transform", to_magnitude_spectrum(pr3))
    plt.show()
    
    
    

