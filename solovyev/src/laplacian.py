import cv2
import numpy as np

k = 4
test_img_path = "../img/V/{task}/test{task}.jpg"
new_img_path  = "../img/V/{task}/_{name}.jpg"
test_img_path = "../img/V/{task}/skel.tif"

def save_jpg(name, img):
    cv2.imwrite(new_img_path.format(task=k, name=name), img)

def blur(img, kk=5):
    res =  cv2.blur(img, (kk, kk))
    print "blur", res.dtype
    return res
    
def super_enhancement(img):
    res = blur(grad_enhancement(img), kk=13) * my_laplacian(img)
    print "super_enhancement", res.dtype
    return res

def laplacian_spatial(img):
    res = cv2.Laplacian(img, cv2.CV_64F)
    return res

def grad_enhancement(img):
    gy = grad_y(img)
    gx = grad_x(img)
    res = gx + gy + img
    print "ge", res.dtype
    return res
    

def grad_x(img, kk=3):
    return cv2.Sobel(img,cv2.CV_64F, 1, 0, ksize=kk)

def grad_y(img, kk=3):
    return cv2.Sobel(img,cv2.CV_64F, 0, 1, ksize=kk)

def my_laplacian(img):
    mask = np.array([[0,  1, 0],
                     [1, -4, 1],
                     [0,  1, 0]])
    return cv2.filter2D(img, -1, mask)


def laplacian_enhancement(img, A=1):
    return A * img - laplacian_spatial(img)



img = cv2.imread(test_img_path.format(task=k), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
img = np.array(img, np.uint16)
save_jpg("laplaced1",    laplacian_enhancement(img, A=0))
save_jpg("laplaced1_5",  laplacian_enhancement(img, A=1.5))
save_jpg("laplaced1_75", laplacian_enhancement(img, A=1.75))
save_jpg("my_laplaced", my_laplacian(img))
print "done with laplacians"

save_jpg("gx",    grad_x(img))
save_jpg("gy",    grad_y(img))
save_jpg("ge",    grad_enhancement(img))
save_jpg("super", super_enhancement(img))
print "done with rest"



