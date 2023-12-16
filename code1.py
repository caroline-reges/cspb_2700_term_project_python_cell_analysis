#!/usr/bin/env python
import cv2
import numpy
import math
import pandas
import matplotlib.pyplot as plt 


#load macrophage and fibroblast image (the 0 makes it greyscale)
mac = cv2.imread("Image_CH3_Z000.png", 0)
fib = cv2.imread("Image_CH4_Z000.png", 0)


#threshold both images. INV means the resulting image will be inverse
ret,threshMac = cv2.threshold(mac, 80,255, cv2.THRESH_BINARY_INV)
ret,threshFib = cv2.threshold(fib, 80, 255, cv2.THRESH_BINARY_INV)


#save results
cv2.imwrite("thresholdedMac.png", threshMac)
cv2.imwrite("thresholdedFib.png", threshFib)

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
keypoint_info_mac = detectorobj.detect(threshMac)
keypoint_info_fib = detectorobj.detect(threshFib)

#draw circles around keypoints
output_image_mac = cv2.drawKeypoints(threshMac, keypoint_info_mac, 0, (0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS) 
output_image_fib = cv2.drawKeypoints(threshFib, keypoint_info_fib, 0, (0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS) 

#save result
cv2.imwrite("circledMac.png", output_image_mac)
cv2.imwrite("circledFib.png", output_image_fib)
#confirm that the images look decent and make sense before moving on!!

#create a dictionary of lists, one for each cell, that contains info about each cell
mac_dict = {}
i = 0
for keyPoint in keypoint_info_mac:
    i += 1
    mac_dict[i] = [keyPoint.pt[0], keyPoint.pt[1], keyPoint.size, "No", 0] #x-axis coordinate, y-axis coordinate, diameter, close to a fibroblast, average inertia

#create another dictionary of fibroblasts, but we only care about x and y axis coordinates
fib_dict = {}
i = 0
for keyPoint in keypoint_info_fib:
    i += 1
    fib_dict[i] = [keyPoint.pt[0], keyPoint.pt[1]] #x-axis coordinate, y-axis coordinate

#Alter the macrophage dictionary entry if they are within a certain distance of a fibroblast
for mackey in mac_dict:
    #if the macrophage coordinates are within a certain distance of any of the fibroblasts
    for fibkey in fib_dict:
        if(math.sqrt((fib_dict[fibkey][0] - mac_dict[mackey][0])**2 + (fib_dict[fibkey][1] - mac_dict[mackey][1])**2) < 100):  #sqrt((x2-x1)^2 + (y2-y1)^2) < 100
            #change "No" to "Yes"
            mac_dict[mackey][3] = "Yes"

#Determine inertia of each macrophage by running simple blob detector over and over
minVal = 0.00001
maxVal = 0.01
approxInertia = 0.01

for x in range(99):
    #set blob detector parameters
    params2 = cv2.SimpleBlobDetector_Params()

    params2.minArea = 100
    params2.maxArea = 5000

    params2.filterByCircularity = False
    params2.filterByConvexity = False
    params2.filterByInertia = True
    params2.minInertiaRatio = minVal
    params2.maxInertiaRatio = maxVal

    #Set up default detector
    detectorobj2 = cv2.SimpleBlobDetector_create(params2)

    #Detect blobs
    keypoint_info_mac2 = detectorobj2.detect(threshMac)

    #for each keypoint in this group of keypoints
    for keypoint in keypoint_info_mac2:
        for mackey in mac_dict:
            if keypoint.pt[0] == mac_dict[mackey][0]:
                mac_dict[mackey][4] = approxInertia

    #increment values
    minVal += 0.01
    maxVal += 0.01
    approxInertia += 0.01

#collect data, turn into data frame
i = 0
data = []
for mackey in mac_dict:
    item = mac_dict[mackey] #x-axis coordinate, y-axis coordinate, diameter, close to a fibroblast, average inertia
    data.append(item)

df = pandas.DataFrame(data, columns=['X-Coordinate', 'Y-Coordinate', 'Diameter', 'Close to Fibroblast', 'Inertia'])

#export data frame as csv
df.to_csv('SimpleBlobData.csv')

#calculate means and sd of each group
yesList = []
noList = []

for mackey in mac_dict:
    if mac_dict[mackey][3] == 'Yes':
        yesList.append(mac_dict[mackey][4])
    else:
        noList.append(mac_dict[mackey][4])

yesMean = numpy.mean(yesList)
noMean = numpy.mean(noList)

yesStd = numpy.std(yesList)
noStd = numpy.std(noList)


#graph
x = ['Near', 'Far']
y = [yesMean, noMean]
yerr = [yesStd, noStd]

plt.figure(figsize=(5, 5))
plt.bar(x, y)
plt.xlabel('Near or Far From Fibroblast')
plt.ylabel('Inertia')
plt.errorbar(x, y, yerr,fmt='.', ecolor='Black')
plt.savefig('barplot.png')


#compare with cell profiler data






