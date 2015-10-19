'''
Created on 28.11.2014

@author: Vladislav
'''
import cv2
import numpy
from scipy import linalg
from matplotlib import pyplot as plt

if __name__ == '__main__':
    image_original = cv2.imread("mandril.bmp", cv2.IMREAD_GRAYSCALE)
    rows = image_original.shape[0]
    columns = image_original.shape[1]
    matrix_for_rotation = cv2.getRotationMatrix2D(center=(rows/2, columns/2), angle=45.0, scale=0.5)
    ROTATION = numpy.mat(matrix_for_rotation)[ : , :-1]
    TRANSLOCATION = numpy.mat(matrix_for_rotation)[ : , -1:]
    image_rotated =cv2.warpAffine(image_original, matrix_for_rotation, (rows, columns))
    sift_detector = cv2.SIFT()
    print "Processing..."
    keypoints_original, descriptors_original = sift_detector.detectAndCompute(image_original, None)
    keypoints_rotated, descriptors_rotated = sift_detector.detectAndCompute(image_rotated, None)
    image_original_with_keypoints = cv2.drawKeypoints(image_original, keypoints_original, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image_rotated_with_keypoints = cv2.drawKeypoints(image_rotated, keypoints_rotated, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    brute_forcer = cv2.BFMatcher()
    match_objects = brute_forcer.match(descriptors_original, descriptors_rotated)
    keypoints_original_good = []
    keypoints_rotated_good = []
    threshold_value = 2.0
    for match_object in match_objects:
        original_point = keypoints_original[match_object.queryIdx].pt
        rotated_point = keypoints_rotated[match_object.trainIdx].pt
        U = numpy.mat(original_point).transpose()
        V = numpy.mat(rotated_point).transpose()
        U_TRANSFORMED = ROTATION * U + TRANSLOCATION
        DELTA_VECTOR = numpy.squeeze(numpy.asarray(V - U_TRANSFORMED))
        # / r1 r2 t1 \ / u1 \  - / v1 \
        # \ r3 r4 t2 / \ u2 /  - \ v2 / 
        if linalg.norm(DELTA_VECTOR) < threshold_value:
            keypoints_original_good.append(keypoints_original[match_object.queryIdx])
            keypoints_rotated_good.append(keypoints_rotated[match_object.trainIdx])
    image_original_with_good_points = cv2.drawKeypoints(image_original, keypoints_original_good)
    image_rotated_with_good_points = cv2.drawKeypoints(image_rotated, keypoints_rotated_good)
    print 'Part of well detected points is ', (float(len(keypoints_original_good)) / len(keypoints_original) * 100), "%"
    plt.subplot(321),plt.imshow(image_original, cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(322),plt.imshow(image_rotated, cmap = 'gray')
    plt.title('Rotated Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(323),plt.imshow(image_original_with_keypoints, cmap = 'gray')
    plt.title('Original With KPs'), plt.xticks([]), plt.yticks([])
    plt.subplot(324),plt.imshow(image_rotated_with_keypoints, cmap = 'gray')
    plt.title('Rotated With KPs'), plt.xticks([]), plt.yticks([])
    plt.subplot(325),plt.imshow(image_original_with_good_points, cmap = 'gray')
    plt.title('Original With KPs Matched'), plt.xticks([]), plt.yticks([])
    plt.subplot(326),plt.imshow(image_rotated_with_good_points, cmap = 'gray')
    plt.title('Rotated With KPs Matched'), plt.xticks([]), plt.yticks([])
    plt.show()
    cv2.waitKey(0)