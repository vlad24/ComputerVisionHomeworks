'''
Created on Apr 16, 2017

@author: vlad
'''

import cv2
from matplotlib import pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.nan,linewidth=np.nan)

k = 7

test_img1_path =  "../img/V/{task}/test{task}.jpg"
test_img2_path = "../img/V/{task}/test{task}.png"
speed_img_path = "../img/V/{task}/speed{task}.jpg"
new_img_path =   "../img/V/{task}/_{name}.jpg"


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
    h                 = get_homomorphic_mask(img.shape, g1, g2, D, c)
    loged             = log_img(img)
    loged_ft          = fourier_transform(loged)
    loged_ft_filtered = multiply_images(loged_ft, h)
    loged_filtered    = fourier_transform_inverse(loged_ft_filtered)
    exped             = normalize(np.exp(loged_filtered))
    return (exped, h)

def equalization(img):
    hist = np.zeros(256, dtype=np.float64)
    result = np.zeros(img.shape, dtype=np.float64)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            T = int(img[i][j])
            hist[T] += 1
    hist = hist / (img.shape[0] * img.shape[1])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            T = int(img[i][j])
            result[i][j] = sum(hist[0:T])
    return np.uint8(result * 255)


if __name__ == '__main__':
    method_demo_needed  = not True
    speed_test_needed   = True
    g1 = 0.5
    g2 = 2.0
    D  = 256
    c  = 1.0
    if method_demo_needed:
        ####
        img1 = cv2.imread(test_img1_path.format(task=k), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(test_img2_path.format(task=k), cv2.IMREAD_GRAYSCALE)
        ft1  = fourier_transform(img1)
        ft2  = fourier_transform(img2)
        ht1, mask = homomorphic_transform(img1, g1, g2, D, c)
        ht2, _    = homomorphic_transform(img2, g1, g2, D, c)
        eq1 = equalization(normalize(ht1))
        eq2 = equalization(normalize(ht2))
        ####
        rows = 3
        cols = 3
        plot_part(rows, cols, 1, "Original 1",              img1)
        plot_part(rows, cols, 4, "Fourier transform 1",     safe_log(to_magnitude_spectrum(ft1)))
        plot_part(rows, cols, 7, "Homomorphic transform 1", eq1)
        plot_part(rows, cols, 5, "H(u,v)",                  mask)
        plot_part(rows, cols, 3, "Original 2",              img2)
        plot_part(rows, cols, 6, "Fourier transform 2",     safe_log(to_magnitude_spectrum(ft2)))
        plot_part(rows, cols, 9, "Homomorphic transform 2", eq2)
        plt.show()
            
    if speed_test_needed:
        img1 = cv2.imread(test_img1_path.format(task=k), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(speed_img_path.format(task=k), cv2.IMREAD_GRAYSCALE)
        ht1 = homomorphic_transform(img1, g1, g2, D, c)        
        ht2 = homomorphic_transform(img2, g1, g2, D, c)        
        #Measure time