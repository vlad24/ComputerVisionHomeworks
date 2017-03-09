import numpy as np
import cv2

def gamma_correction(img, correction):
    print "Gamma"
    result = img / 255.0
    result = cv2.pow(result, correction)
    return np.uint8(result * 255)


def log_correction(img):
    print "Log"
    b = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b[i][j] = np.log(1.0 + img[i][j] / 255.0)
    return np.uint8(b * 255)

def neg_correction(img):
    print "Negative"
    b = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b[i][j] = 1.0 - img[i][j] / 255.0
    return np.uint8(b * 255)

image = cv2.imread('kitty.jpg', cv2.IMREAD_GRAYSCALE)
alpha = 0.5
gamma_image = gamma_correction(image, alpha)
log_image = log_correction(image)
neg_image = neg_correction(image)

cv2.imshow("Gamma {}".format(alpha), gamma_image)
cv2.waitKey()
cv2.imshow("Logarithm result", log_image)
cv2.waitKey()
cv2.imshow("Negative result", neg_image)
cv2.waitKey()
