'''
Created on 16.11.2014
@author: Vladislav
'''

import cv2
import numpy as np

color = (0, 0, 0) ; thickness = 1; parent_index = 3; absent = -1 

if __name__ == '__main__':
    original_image = cv2.imread("text.bmp", cv2.IMREAD_GRAYSCALE)
    if original_image is not None:
        cv2.imshow("Original Image", original_image)
        image = cv2.GaussianBlur(src=original_image, ksize=(3,3), sigmaX=0)
        image = cv2.Laplacian(src=image, ddepth=-1, ksize=3)
        structural_element = np.ones((2, 3), np.uint8)
        image = cv2.dilate(image, structural_element, iterations=2)
#         cv2.imshow("Dilate. ksize = 2x3", image)
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]
#         height, width = image.shape
#         mask = np.zeros((height + 2, width + 2), np.uint8)
#         for contour_number in range(len(contours)):
#             #drawing rectangle
#                 cv2.drawContours(mask, [contours[contour_number]], 0, (255, 255, 255), 0)
#         cv2.floodFill(image, mask, (0, 0), (255, 255, 255))
#         cv2.imshow("Flood filled", 255-image)
        for contour_number in range(len(contours)):
            #drawing rectangles
            if hierarchy[contour_number][parent_index] == absent:
                top_left_x, top_left_y, width, height = cv2.boundingRect(contours[contour_number])
                top_left_point = (top_left_x, top_left_y)
                bottom_right_point = (top_left_x + width, top_left_y + height)
                cv2.rectangle(original_image, top_left_point, bottom_right_point, color, thickness)
        cv2.imshow("Words detected.", original_image)
        cv2.imwrite('words_detected.bmp', original_image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        cv2.waitKey()
    else:
        raise Exception("No source image found!")
