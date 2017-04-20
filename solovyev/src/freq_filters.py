'''
Created on Apr 14, 2017

@author: vlad
'''

import cv2
from matplotlib import pyplot as plt
import numpy as np

k = 6

test_img_path = "../img/V/{task}/test{task}.jpg"
new_img_path = "../img/V/{task}/_{name}.jpg"


def fourier_transform(img):
    return np.fft.fftshift(np.fft.fft2(img))


def fourier_transform_inverse(fourier_transform):
    return np.abs(np.fft.ifft2(np.fft.ifftshift(fourier_transform)))


def circular_centered_mask(shape, radius):
    angle_range = (0, 360)
    x, y = np.ogrid[:shape[0], :shape[1]]
    cx = shape[0] // 2
    cy = shape[1] // 2
    tmin, tmax = np.deg2rad(angle_range)
    if tmax < tmin:
            tmax += 2 * np.pi
    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    theta = np.arctan2(x - cx, y - cy) - tmin
    theta %= (2 * np.pi)
    circmask = r2 <= radius * radius
    anglemask = theta <= tmax - tmin
    res = (circmask * anglemask).astype(np.uint8)
    return res


def get_mask_4ideal(ft, percentage=50, is_high_pass=False):
    n = min(ft.shape[0], ft.shape[1])
    mag = to_magnitude_spectrum(ft)
    total_sum = sum(sum(mag))
    mask = None
    rl = 1
    rr = n
    r = (rl + rr) // 2
    ratio = 0
    while (rl < rr):
        mask = circular_centered_mask(ft.shape, r)
        current_sum = np.sum(mask * mag)
        ratio = 100.0 * current_sum / total_sum
        if (ratio > percentage):
            rr = r - 1
        elif (ratio < percentage):
            rl = r + 1
        else:
            break
        r = (rl + rr) // 2
    # print r,ratio
    result = circular_centered_mask(img.shape, r - 1)
    if is_high_pass:
        result = 1 - result
    return result


def get_mask_4butterwort(shape, D, n, is_high_pass=False):
    x, y = np.ogrid[ : shape[0], : shape[1]]
    cx = shape[0] // 2
    cy = shape[1] // 2
    D2 = float(D ** 2)
    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    result = 1.0 / (1.0 + np.power((r2 / D2), n))
    if is_high_pass:
        result = 1 - result
    return result


def get_mask_4gauss(shape, D, is_high_pass=False):
    x, y = np.ogrid[ : shape[0], : shape[1]]
    cx = shape[0] // 2
    cy = shape[1] // 2
    D2 = float(D ** 2)
    distance = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    result = np.exp(-distance / (2 * D2))
    if is_high_pass:
        result = 1 - result
    return result


def get_mask_4laplacian_freq(shape):
    x, y = np.ogrid[ : shape[0], : shape[1]]
    cx = shape[0] // 2
    cy = shape[1] // 2
    distance = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    return -distance


def multiply_images(image, mask):
    return image * mask


def to_magnitude_spectrum(t):
    return np.abs(t)


def laplacian_enhancement(img, A=1):
    return A * img - cv2.Laplacian(img, cv2.CV_64F) 

def safe_log(t):
    return np.log(1 + np.abs(t))


def plot_part(rs, cs, num, name, img):
    plt.subplot(rs, cs, num)
    plt.title(name)
    plt.imshow(img, cmap='gray')
    plt.xticks([]); plt.yticks([])


def normalize(img, max_constant=255):
    minimum = np.amin(img)
    maximum = np.amax(img)
    result = max_constant * (img - minimum) / (maximum - minimum) 
    return result

if __name__ == '__main__':
    ideal_demo_needed     = not True
    btw_demo_needed       = not True
    gauss_demo_needed     = not True
    laplacian_demo_needed = True
    ####################################################################################
    img = cv2.imread(test_img_path.format(task=k), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(test_img_path.format(task=k-1), cv2.IMREAD_GRAYSCALE)
    ft = fourier_transform(img)
    ft2 = fourier_transform(img2)
    ###################################IDEAL##############################################
    if ideal_demo_needed:
        rows = 3
        cols = 3
        percentage1 = 80
        percentage2 = 30
        percentage3 = 10
        idealf_mask_p1 = get_mask_4ideal(ft, percentage=percentage1)
        idealf_mask_p2 = get_mask_4ideal(ft, percentage=percentage2)
        idealf_mask_p3 = get_mask_4ideal(ft, percentage=percentage3)
        idealf_ft_p1   = multiply_images(ft, idealf_mask_p1)
        idealf_ft_p2   = multiply_images(ft, idealf_mask_p2)
        idealf_ft_p3   = multiply_images(ft, idealf_mask_p3)
        plot_part(rows, cols, 1, "IDEAL LP Img  per={}%".format(percentage1), fourier_transform_inverse(idealf_ft_p1))
        plot_part(rows, cols, 2, "IDEAL LP FT   per={}%".format(percentage1), safe_log(idealf_ft_p1))
        plot_part(rows, cols, 3, "IDEAL LP Mask per={}%".format(percentage1), idealf_mask_p1)
        plot_part(rows, cols, 4, "IDEAL LP Img  per={}%".format(percentage2), fourier_transform_inverse(idealf_ft_p2))
        plot_part(rows, cols, 5, "IDEAL LP FT   per={}%".format(percentage2), safe_log(idealf_ft_p2))
        plot_part(rows, cols, 6, "IDEAL LP Mask per={}%".format(percentage2), idealf_mask_p2)
        plot_part(rows, cols, 7, "IDEAL LP Img  per={}%".format(percentage3), fourier_transform_inverse(idealf_ft_p3))
        plot_part(rows, cols, 8, "IDEAL LP FT   per={}%".format(percentage3), safe_log(idealf_ft_p3))
        plot_part(rows, cols, 9, "IDEAL LP Mask per={}%".format(percentage3), idealf_mask_p3)
        plt.show()
        
    ######################################BTW############################################
    if btw_demo_needed:
        rows = 4
        cols = 3
        D1 = 15
        D2 = 30
        D3 = 80
        n1 = 1 
        n2 = 3
        btw_mask_D1_n1 = get_mask_4butterwort(ft.shape, D=D1, n=n1)
        btw_mask_D1_n2 = get_mask_4butterwort(ft.shape, D=D2, n=n2)
        btw_mask_D2_n1 = get_mask_4butterwort(ft.shape, D=D2, n=n1)
        btw_mask_D3_n1 = get_mask_4butterwort(ft.shape, D=D3, n=n1)
        btw_ft_D1_n1   = multiply_images(ft, btw_mask_D1_n1)
        btw_ft_D1_n2   = multiply_images(ft, btw_mask_D1_n2)
        btw_ft_D2_n1   = multiply_images(ft, btw_mask_D2_n1)
        btw_ft_D3_n1   = multiply_images(ft, btw_mask_D3_n1)
        plot_part(rows, cols, 1,  "BTW LP Img  D={} n={}".format(D1, 2 * n1), fourier_transform_inverse(btw_ft_D1_n1))
        plot_part(rows, cols, 2,  "BTW LP FT   D={} n={}".format(D1, 2 * n1), safe_log(to_magnitude_spectrum(btw_ft_D1_n1)))
        plot_part(rows, cols, 3,  "BTW LP Mask D={} n={}".format(D1, 2 * n1), btw_mask_D1_n1)
        plot_part(rows, cols, 4,  "BTW LP Img  D={} n={}".format(D1, 2 * n2), fourier_transform_inverse(btw_ft_D1_n2))
        plot_part(rows, cols, 5,  "BTW LP FT   D={} n={}".format(D1, 2 * n2), safe_log(to_magnitude_spectrum(btw_ft_D1_n2)))
        plot_part(rows, cols, 6,  "BTW LP Mask D={} n={}".format(D1, 2 * n2), btw_mask_D1_n2)
        plot_part(rows, cols, 7,  "BTW LP Img  D={} n={}".format(D2, 2 * n1), fourier_transform_inverse(btw_ft_D2_n1))
        plot_part(rows, cols, 8,  "BTW LP FT   D={} n={}".format(D2, 2 * n1), safe_log(to_magnitude_spectrum(btw_ft_D2_n1)))
        plot_part(rows, cols, 9,  "BTW LP Mask D={} n={}".format(D2, 2 * n1), btw_mask_D2_n1)
        plot_part(rows, cols, 10, "BTW LP Img  D={} n={}".format(D3, 2 * n1), fourier_transform_inverse(btw_ft_D3_n1))
        plot_part(rows, cols, 11, "BTW LP FT   D={} n={}".format(D3, 2 * n1), safe_log(to_magnitude_spectrum(btw_ft_D3_n1)))
        plot_part(rows, cols, 12, "BTW LP Mask D={} n={}".format(D3, 2 * n1), btw_mask_D3_n1)
        plt.show()
        
    #######################################GAUSS##########################################
    if gauss_demo_needed:
        rows = 3
        cols = 3
        D1 = 15
        D2 = 30
        D3 = 80
        gauss_mask_D1 = get_mask_4gauss(ft.shape, D=D1)
        gauss_mask_D2 = get_mask_4gauss(ft.shape, D=D2)
        gauss_mask_D3 = get_mask_4gauss(ft.shape, D=D3)
        gauss_ft_D1 = multiply_images(ft, gauss_mask_D1)
        gauss_ft_D2 = multiply_images(ft, gauss_mask_D2)
        gauss_ft_D3 = multiply_images(ft, gauss_mask_D3)
        plot_part(rows, cols, 1, "GAUSS LP Img  D={}".format(D1), fourier_transform_inverse(gauss_ft_D1))
        plot_part(rows, cols, 2, "GAUSS LP FT   D={}".format(D1), safe_log(to_magnitude_spectrum(gauss_ft_D1)))
        plot_part(rows, cols, 3, "GAUSS LP Mask D={}".format(D1), gauss_mask_D1)
        plot_part(rows, cols, 4, "GAUSS LP Img  D={}".format(D2), fourier_transform_inverse(gauss_ft_D2))
        plot_part(rows, cols, 5, "GAUSS LP FT   D={}".format(D2), safe_log(to_magnitude_spectrum(gauss_ft_D2)))
        plot_part(rows, cols, 6, "GAUSS LP Mask D={}".format(D2), gauss_mask_D2)
        plot_part(rows, cols, 7, "GAUSS LP Img  D={}".format(D3), fourier_transform_inverse(gauss_ft_D3))
        plot_part(rows, cols, 8, "GAUSS LP FT   D={}".format(D3), safe_log(to_magnitude_spectrum(gauss_ft_D3)))
        plot_part(rows, cols, 9, "GAUSS LP Mask D={}".format(D3), gauss_mask_D3)
        plt.show()
        ###High Pass
        gauss_mask_hp_D1 = 1 - gauss_mask_D1
        gauss_mask_hp_D2 = 1 - gauss_mask_D2
        gauss_mask_hp_D3 = 1 - gauss_mask_D3
        gauss_ft_hp_D1   = multiply_images(ft, gauss_mask_hp_D1)
        gauss_ft_hp_D2   = multiply_images(ft, gauss_mask_hp_D2)
        gauss_ft_hp_D3   = multiply_images(ft, gauss_mask_hp_D3)
        plot_part(rows, cols, 1, "GAUSS HP Img  D={}".format(D1), fourier_transform_inverse(gauss_ft_hp_D1))
        plot_part(rows, cols, 2, "GAUSS HP FT   D={}".format(D1), safe_log(to_magnitude_spectrum(gauss_ft_hp_D1)))
        plot_part(rows, cols, 3, "GAUSS HP Mask D={}".format(D1), gauss_mask_hp_D1)
        plot_part(rows, cols, 4, "GAUSS HP Img  D={}".format(D2), fourier_transform_inverse(gauss_ft_hp_D2))
        plot_part(rows, cols, 5, "GAUSS HP FT   D={}".format(D2), safe_log(to_magnitude_spectrum(gauss_ft_hp_D2)))
        plot_part(rows, cols, 6, "GAUSS HP Mask D={}".format(D2), gauss_mask_hp_D2)
        plot_part(rows, cols, 7, "GAUSS HP Img  D={}".format(D3), fourier_transform_inverse(gauss_ft_hp_D3))
        plot_part(rows, cols, 8, "GAUSS HP FT   D={}".format(D3), safe_log(to_magnitude_spectrum(gauss_ft_hp_D3)))
        plot_part(rows, cols, 9, "GAUSS HP Mask D={}".format(D3), gauss_mask_hp_D3)
        plt.show()
    ######################################LAPLACIAN############################################
    if laplacian_demo_needed:
        rows = 2
        cols = 3
        laplacian_freq_mask = get_mask_4laplacian_freq(ft2.shape)
        laplaced_ft         = multiply_images(ft2, laplacian_freq_mask)
        laplacian_result    = fourier_transform_inverse(laplaced_ft)
        plot_part(rows, cols, 1, "ORIGINAL",         img2)
        plot_part(rows, cols, 4, "FT",               safe_log(to_magnitude_spectrum(ft2)))
        plot_part(rows, cols, 2, "LAPLACIAN",        normalize(laplacian_result))
        plot_part(rows, cols, 5, "LAPLACED FT ",     safe_log(to_magnitude_spectrum(laplaced_ft)))
        plot_part(rows, cols, 3, "COMBO",            img2 - normalize(laplacian_result))
        plt.show()
        
    
    
    
        
    
    