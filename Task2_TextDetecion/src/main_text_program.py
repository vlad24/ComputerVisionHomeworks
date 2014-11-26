'''
Created on 06.11.2014

@author: Vladislav
'''

import cv2

if __name__ == '__main__':
    image = cv2.imread("text.bmp", cv2.IMREAD_GRAYSCALE)
    if image is not None:
        cv2.imshow("Original Image", image)
        image = cv2.GaussianBlur(src=image, ksize=(3,3), sigmaX=0)
        cv2.imshow("Gaussian Blur. ksize = 3. s_x = 0", image)
        image = cv2.Laplacian(src=image, ddepth=-1, ksize=3)
        cv2.imshow("Laplacian operator. ksize = 3", image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        cv2.waitKey()
    else:
        raise Exception("No source image found!")
