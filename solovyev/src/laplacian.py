import cv2

k = 4
test_img_path = "../img/V/{task}/test{task}.jpg"
new_img_path  = "../img/V/{task}/_{name}.jpg"
#TODO remove
test_img_path = "../img/V/{task}/skel.tif"

def save_jpg(name, img):
    cv2.imwrite(new_img_path.format(task=k, name=name), img)

def super_enhancement(img):
    return grad_enhancement(laplacian_enhancement(cv2.blur(img, (5,5))))

def ultra_enhancement(img):
    return grad_enhancement(img) * laplacian(img)

def laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F)

def laplacian_enhancement(img, A=1):
    return A * img - laplacian(img)

def grad_enhancement(img):
    gy = grad_y(img)
    gx = grad_x(img)
    return gx + gy + img

def grad_x(img, kk=3):
    return cv2.Sobel(img,cv2.CV_64F, 1, 0, ksize=kk)

def grad_y(img, kk=3):
    return cv2.Sobel(img,cv2.CV_64F, 0, 1, ksize=kk)



img = cv2.imread(test_img_path.format(task=k), cv2.IMREAD_GRAYSCALE)
save_jpg("laplaced1",    laplacian_enhancement(img, A=1))
save_jpg("lablaced1_5",  laplacian_enhancement(img, A=1.5))
save_jpg("lablaced1_75", laplacian_enhancement(img, A=1.75))
print "done with laplacians"

save_jpg("gx",    grad_x(img))
save_jpg("gy",    grad_y(img))
save_jpg("ge",    grad_enhancement(img))
save_jpg("super", super_enhancement(img))
save_jpg("ultra", ultra_enhancement(img))
print "done with grads"



