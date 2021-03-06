import cv2
import numpy as np
import time

def filter_image(img, hsv_lower, hsv_upper):
#    img_filt = cv2.medianBlur(img, 5)
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    imgToBlur = cv2.medianBlur(img, 5)
    imagehsv = cv2.cvtColor(imgToBlur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imagehsv, hsv_lower, hsv_upper)
    return mask

    ###############################################################################
    ### You might need to change the parameter values to get better results
    ###############################################################################
def detect_blob(mask):
     img = cv2.medianBlur(mask, 11)
    
     # Set up the SimpleBlobdetector with default parameters with specific values.
     params = cv2.SimpleBlobDetector_Params()
    
     _, threshold = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

     params.minThreshold = 10
     params.maxThreshold = 255
     params.filterByArea = True
     params.minArea = 200
     params.filterByInertia = False
     params.filterByConvexity = False
    
     # builds a blob detector with the given parameters
     detector = cv2.SimpleBlobDetector_create(params)
    
     # use the detector to detect blobs.
     keypoints = detector.detect(threshold)
     return keypoints

def find_cube(img, hsv_lower, hsv_upper):
    """Find the cube in an image.
        Arguments:
        img -- the image
        hsv_lower -- the h, s, and v lower bounds
        hsv_upper -- the h, s, and v upper bounds
        Returns [x, y, radius] of the target blob, and [0,0,0] or None if no blob is found.
    """
    mask = filter_image(img, hsv_lower, hsv_upper)
    keypoints = detect_blob(mask)

    if keypoints == []:
        return None
    
    ###############################################################################
    # Todo: Sort the keypoints in a certain way if multiple key points get returned
    ###############################################################################

    #keypoints = sorted(keypoints, key=lambda keypoint: keypoint.size, reverse=True)
    return [keypoints[0].pt[0], keypoints[0].pt[1], keypoints[0].size]

