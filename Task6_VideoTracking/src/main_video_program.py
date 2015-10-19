'''
Created on 16.12.2014

@author: Vladislav
'''

import cv2
import numpy as np


def track_Harris_feature_points(video_name, color, writeFlag):
    global name
    name = "Harris"
    video = cv2.VideoCapture(video_name)
    detector = lambda img: get_Harris_points(img)
    points_tracking(video, detector, color, writeFlag)
    video.release()

def track_FAST_feature_points(video_name, color, writeFlag):
    global name
    name = "FAST"
    video = cv2.VideoCapture(video_name)
    detector = lambda img : get_FAST_points(img)
    points_tracking(video, detector, color, writeFlag)
    video.release()

def get_Harris_points(gray_img):
    harris = cv2.cornerHarris(gray_img, 2, 3, 0.05)
    x_array, y_array = np.nonzero(harris > 0.03 * harris.max())
    harris_points_array = [np.array([[x,y]]) for x,y in zip(x_array,y_array)]
    return np.array(harris_points_array, dtype=np.float32)

def get_FAST_points(gray_img):
    fast = cv2.FastFeatureDetector_create(threshold=50)
    kp_array = fast.detect(gray_img,None)
    fastPointsArray = [np.array([[kp.pt[0], kp.pt[1]]]) for kp in kp_array]
    return np.array(fastPointsArray, dtype=np.float32)

def get_good_next_points(frame, old_gray, oldTrackPoints):
    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    win_size = 15
    lucas_kanade_params = dict( winSize  = (win_size,win_size), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    new_track_points, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, oldTrackPoints, None, **lucas_kanade_params)
    new_points = new_track_points[st==1]
    old_points = oldTrackPoints[st==1]
    return (new_track_points, new_points, old_points)

def draw_tracks(mask, frame, new_points, old_points):
    for j, (new, old) in enumerate(zip(new_points,old_points)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b), (c,d), color[j % len(color)].tolist(), 2)
                frame = cv2.circle(frame, (a,b), 2, color[j % len(color)].tolist(),-1)
    track_img = cv2.add(frame, mask)
    return (mask,track_img)

def points_tracking(video, detector, color, writeFlag):
    ret, old_frame = video.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(old_frame)
    if writeFlag:
        height, width, _ = old_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_video = cv2.VideoWriter('points_tracking'+name+'.avi',fourcc, 10, (width,height))
    oldTrackPoints = detector(old_gray)
    while(video.isOpened()):
        ret, frame = video.read()
        if ret:
            new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            new_track_points, new_points, old_points = get_good_next_points(frame, old_gray, oldTrackPoints)
            mask, tracking_image = draw_tracks(mask, frame, new_points, old_points)
            cv2.imshow('feature Points Tracking '+name, tracking_image)
            if writeFlag:
                out_video.write(tracking_image)
            cv2.waitKey(10)
            oldTrackPoints = new_points.reshape(-1,1,2)
            old_gray = new_gray.copy()
        else:
            break

if __name__ == '__main__':
    color = np.random.randint(0,255,(100,3))
    track_Harris_feature_points('sequence.mpg', color, True)
    cv2.destroyAllWindows()