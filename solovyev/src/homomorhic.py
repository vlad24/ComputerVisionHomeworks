'''
Created on Apr 16, 2017

@author: vlad
'''

import cv2
from matplotlib import pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.nan,linewidth=np.nan)

k = 7

test_img_path = "../img/V/{task}/test{task}.png"
new_img_path = "../img/V/{task}/_{name}.jpg"


def fourier_transform(img):
    return np.fft.fftshift(np.fft.fft2(img))


def fourier_transform_inverse(fourier_transform):
    return np.abs(np.fft.ifft2(np.fft.ifftshift(fourier_transform)))


def to_magnitude_spectrum(t):
    return np.abs(t)


def safe_log(t):
    return np.log(1 + np.abs(t))


def normalize(img, max_constant=255, min_constant=0):
    minimum = np.amin(img)
    maximum = np.amax(img)
    result = max_constant * ((img - minimum) / (maximum - minimum) + min_constant)
    return result


def multiply_images(image, mask):
    return image * mask


def plot_part(rs, cs, num, name, img, vmin=None, vmax=None):
    plt.subplot(rs, cs, num)
    plt.title(name)
    plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    plt.xticks([]); plt.yticks([])

def get_homomorphic_mask(shape, g1, g2, D, c):
    assert (g2 > 1 > g1 > 0)
    x, y = np.ogrid[ : shape[0], : shape[1]]
    cx, cy = shape[0] // 2, shape[1] // 2
    D2 = float(D ** 2)
    distance = np.array((x - cx) * (x - cx) + (y - cy) * (y - cy)).astype(np.float64)
    result =  g1 + (g2 - g1) * (1.0 - np.exp(-c * distance / D2))
    return result

def log_img(img):
    return np.log(img.clip(min=1E-5))

def homomorphic_transform(img, g1, g2, D, c):
    h = get_homomorphic_mask(img.shape, g1, g2, D, c)
    loged             = log_img(img)
    loged_ft          = fourier_transform(loged)
    loged_ft_filtered = multiply_images(loged_ft, h)
    loged_filtered    = fourier_transform_inverse(loged_ft_filtered)
    exped             = normalize(np.exp(loged_filtered))
    return (exped, h, loged, loged_filtered)
    

if __name__ == '__main__':
    g1 = 0.5
    g2 = 2.0
    D  = 256
    c  = 1.0
    ####
    img = cv2.imread(test_img_path.format(task=k), cv2.IMREAD_GRAYSCALE)
    ft  = fourier_transform(img)
    ht, mask, l, ll = homomorphic_transform(img, g1, g2, D, c)
    ####
    rows = 3
    cols = 3
    plot_part(rows, cols, 1, "Original",                 img)
    plot_part(rows, cols, 3, "Fourier transform",        safe_log(to_magnitude_spectrum(ft)))
    plot_part(rows, cols, 4, "Homomorphic transform",    normalize(ht))
    plot_part(rows, cols, 5, "H(u,v)",                   mask, 0, g2)
    plot_part(rows, cols, 6, "Homomorphic transform FT", safe_log(to_magnitude_spectrum(fourier_transform(ht))))
    plot_part(rows, cols, 7, "Loged",                    l, 0, 10)
    plot_part(rows, cols, 9, "Loged filtered",           ll,0, 10)
    plt.show()
        
    
    
    
        
    
