'''
Created on 19.11.2014
@author: Vladislav
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    image = cv2.imread('mandril.bmp', cv2.IMREAD_GRAYSCALE)
    fourier_transform = np.fft.fft2(image)
    fourier_transform_centered = np.fft.fftshift(fourier_transform)
    magnitude_spectrum = 20 * np.log(np.abs(fourier_transform_centered))
    rows_center = image.shape[0] / 2
    columns_center = image.shape[1] / 2
    fourier_transform_centered[rows_center - 30 : rows_center + 30, columns_center - 30 : columns_center + 30] = 0
    f_inverse_shift = np.fft.ifftshift(fourier_transform_centered)
    image_high_pass = np.abs(np.fft.ifft2(f_inverse_shift))
    image_Laplacian = cv2.Laplacian(src=image, ddepth=cv2.CV_8UC1, ksize=1)
    
    plt.subplot(131),plt.imshow(image, cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(image_high_pass, cmap = 'gray')
    plt.title('High Pass Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(image_Laplacian, cmap = 'gray')
    plt.title('Laplaced Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    
    cv2.imwrite('laplaced.bmp',image_Laplacian)
    cv2.imwrite('fouriered.bmp',image_high_pass)
