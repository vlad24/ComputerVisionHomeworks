'''
Created on Apr 14, 2017

@author: vlad
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

k = 5

test_img_path = "../img/V/{task}/test{task}.jpg"
new_img_path  = "../img/V/{task}/_{name}.jpg"


def fourier_transform(img):
    return np.fft.fftshift(np.fft.fft2(img))

def fourier_transform_inverse(fourier_transform):
    return np.abs(np.fft.ifft2(np.fft.ifftshift(fourier_transform)))

def zerofy(fourier_transform, y_offset, x_offset):
    high_pass_img = np.copy(fourier_transform)
    row_c, col_c = (high_pass_img.shape[0] / 2, high_pass_img.shape[1] / 2) 
    if (x_offset == 0 and y_offset == 0):
        high_pass_img[row_c, col_c] = 0
    else:
        high_pass_img[row_c - y_offset : row_c + y_offset + 1, col_c - x_offset : col_c + x_offset + 1] = 0
    return high_pass_img
    

def to_magnitude_spectrum(fourier_transform):
    return 20 * np.log(np.abs(fourier_transform))

def plot_part(part, name, img):
    plt.subplot(part)
    plt.title(name)
    plt.imshow(img, cmap = 'gray')
    plt.xticks([]); plt.yticks([])

if __name__ == '__main__':
    img1 = cv2.imread(test_img_path.format(task=k), cv2.IMREAD_GRAYSCALE)
    ft = fourier_transform(img1)
    hp = zerofy(ft, 30, 30)
    plot_part(221, "Original",             img1)
    plot_part(222, "Fourier Transform",    to_magnitude_spectrum(ft))
    plot_part(223, "High Pass Image",      fourier_transform_inverse(hp))
    plot_part(224, "HP Fourier Transform", to_magnitude_spectrum(hp))
    plt.show()
    
    img2 = cv2.imread(test_img_path.format(task=k-1), cv2.IMREAD_GRAYSCALE)
    ft = fourier_transform(img2)
    hp = zerofy(ft, 30, 30)
    plot_part(221, "Original",             img2)
    plot_part(222, "Fourier Transform",    to_magnitude_spectrum(ft))
    plot_part(223, "High Pass Image",      fourier_transform_inverse(hp))
    plot_part(224, "HP Fourier Transform", to_magnitude_spectrum(hp))
    plt.show()
    
    img3 = cv2.imread(test_img_path.format(task=k-3), cv2.IMREAD_GRAYSCALE)
    ft = fourier_transform(img3)
    hp = zerofy(ft, 0, 0)
    plot_part(221, "Original",             img3)
    plot_part(222, "Fourier Transform",    to_magnitude_spectrum(ft))
    plot_part(223, "High Pass Image",      fourier_transform_inverse(hp))
    plot_part(224, "HP Fourier Transform", to_magnitude_spectrum(hp))
    plt.show()

