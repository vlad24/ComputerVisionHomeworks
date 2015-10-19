'''
Created on Dec 14, 2014

@author: Vladisslav
'''

import cv2
import numpy as np
# Parameters for lucas kanade optical flow
lk_params = dict( winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def getHarrisPoints(gray_img):
    harris = cv2.cornerHarris(gray_img, 2, 3, 0.05)
    # Threshold for an optimal value
    xarr,yarr = np.nonzero(harris > 0.03*harris.max())
    harrisPointsArray = [np.array([[x,y]]) for x,y in zip(xarr,yarr)]
    return np.array(harrisPointsArray, dtype=np.float32)

def getFASTPoints(gray_img):
    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector(threshold=50)
    # find keypoints
    kp_array = fast.detect(gray_img,None)
    # print kp_array[0].pt[1]
    fastPointsArray = [np.array([[kp.pt[0], kp.pt[1]]]) for kp in kp_array]
    return np.array(fastPointsArray, dtype=np.float32)

def videoTracking(cap, slow_motion_delay, detector):
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    error_occured, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    # Define the codec and create VideoWriter object
    height, width, _ = old_frame.shape
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    out = cv2.VideoWriter('output_'+detector+'.avi',fourcc, slow_motion_delay, (width,height))
    if detector == 'Harris':
        oldTrackPoints = getHarrisPoints(old_gray)
    elif detector == 'FAST':
        oldTrackPoints = getFASTPoints(old_gray)
    else:
        print('Not correct method')
    while(cap.isOpened()):
        error_occured, frame = cap.read()
        if error_occured:
            new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            newTrackPoints, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, oldTrackPoints, None, **lk_params)
            # Select good points
            good_new = newTrackPoints[st==1]
            good_old = oldTrackPoints[st==1]
            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), color[i%len(color)].tolist(), 2) #(255,0,0), 2)
                frame = cv2.circle(frame,(a,b),2,color[i%len(color)].tolist(), -1)#(0,255,0),-1)
                track_img = cv2.add(frame, mask)
                #write tracked frame
                out.write(track_img)
                cv2.imshow('videoTracking '+detector, track_img)
                if cv2.waitKey(slow_motion_delay) & 0xFF == ord('q'):
                    break
                # Now update the previous frame and previous points
                old_gray = new_gray.copy()
                oldTrackPoints = good_new.reshape(-1, 1, 2)
        else:
            break
    
if __name__ == '__main__':
    cap = cv2.VideoCapture('sequence.mpg')
    #slow_motion_delay in milliseconds
    slow_motion_delay = 80
    videoTracking(cap, slow_motion_delay, 'FAST')
    cap.release()
    cv2.destroyAllWindows()