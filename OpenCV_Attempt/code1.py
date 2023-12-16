import cv2
import numpy


#load image (the 0 makes it greyscale)
img = cv2.imread("Image_CH3_Z000.png", 0)

#threshold
ret,thresh = cv2.threshold(img, 60,255, cv2.THRESH_BINARY_INV)

#save result
cv2.imwrite("thresholded.png", thresh)

#set blob detector parameters
params = cv2.SimpleBlobDetector_Params()

params.minArea = 100
params.maxArea = 5000

params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False

#Set up default detector
detectorobj = cv2.SimpleBlobDetector_create(params)

#Detect blobs
keypoint_info = detectorobj.detect(thresh)

#draw circles around keypoints
output_image = cv2.drawKeypoints(thresh, keypoint_info, 0, (0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS) 

#save result
cv2.imwrite("circled.png", output_image)

#I can extract the x-y coordinates from the keypoints but that's it. I can't figure out how to get othern data. 
