# some function used for postprocess the labelimg which is generate by the deeplab

import numpy as np
import cv2

# postprocess
# argv: the img labeled by the cnn&densecrf
# return: a processed img(small size) and its area
def regionNMS(img):
    BOUNDING_MERGE_RATIO = 1.15
    BOUNDING_REGION_RATIO = 1.25
    img = img.copy()
    firstMaxImg, firstMaxArea, secondMaxImg, secondMaxArea = basicNMS(img)
    # don't need to process
    if firstMaxArea == 0:     # there is no container
        return img, 0
    if secondMaxArea == 0:    # there is only one region
        return firstMaxImg, firstMaxArea
    
    # judge whether the two region is a container, if true ,return the merged image
    mergeImg = firstMaxImg + secondMaxImg
    top, bottom, farLeft, farRight = boundingRect(mergeImg)

    if (bottom-top)*(farRight-farLeft)/(firstMaxArea+secondMaxArea) < BOUNDING_MERGE_RATIO:   # the two region is a container
        return mergeImg, firstMaxArea+secondMaxArea
    
    # delete one region
    firstTop, firstBottom, firstLeft, firstRight = boundingRect(firstMaxImg)
    firstBoundingAreaRatio = (firstBottom-firstTop) * (firstRight-firstLeft) / firstMaxArea
    secondTop, secondBottom, secondLeft, secondRight = boundingRect(secondMaxImg)
    secondBoundingAreaRatio = (secondBottom-secondTop) * (secondRight-secondLeft) / secondMaxArea
    # if one's shape is good shape and another's shape is not good, choose it. Else, do deeper processing
    if firstBoundingAreaRatio<BOUNDING_REGION_RATIO and secondBoundingAreaRatio>BOUNDING_REGION_RATIO:
        return firstMaxImg, firstMaxArea
    if firstBoundingAreaRatio>BOUNDING_REGION_RATIO and secondBoundingAreaRatio<BOUNDING_REGION_RATIO:
        return secondMaxImg, secondMaxArea
        
    # compute the distance between image center and region center, choose the shorter distance
    firstTop, firstBottom, firstLeft, firstRight = avgRect(firstMaxImg)
    firstX = (firstBottom+firstTop)/2
    firstY = (firstLeft+firstRight)/2
    secondTop, secondBottom, secondLeft, secondRight = avgRect(secondMaxImg)
    secondX = (secondBottom+secondTop)/2
    secondY = (secondLeft+secondRight)/2
    centerX = img.shape[0]/2
    centerY = img.shape[1]/2
    firstDis = (firstX-centerX)**2 + (firstY-centerY)**2
    secondDis = (secondX-centerX)**2 + (secondY-centerY)**2
    if firstDis < secondDis:
        return firstMaxImg, firstMaxArea
    else :
        return secondMaxImg, secondMaxArea
    
# do NMS for the gray label image, hold the two largest region
def basicNMS(img):
    SECOND_FIRST_RATIO = 0.5
    img, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
	# there is no container in image
    if len(contours) == 0:
        return None, 0, None, 0

	# get the two largest area
    firstMaxIdx = 0
    firstMaxArea = 0
    secondMaxIdx = 0
    secondMaxArea = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > firstMaxArea:
            secondMaxIdx = firstMaxIdx
            secondMaxArea = firstMaxArea
            firstMaxIdx = i
            firstMaxArea = area
        elif area > secondMaxArea:
            secondMaxIdx = i
            secondMaxArea = area

    # draw the first largest contour
    firstMaxImg = np.zeros(img.shape)
    firstMaxImg.astype(np.uint8)
    for i in range(len(contours[firstMaxIdx])):
        y = contours[firstMaxIdx][i][0][0]
        x = contours[firstMaxIdx][i][0][1]
        firstMaxImg[x, y] = 255

    # if the second largest contour satisfy the condition, draw it, else return 0
    if SECOND_FIRST_RATIO < secondMaxArea/firstMaxArea:
        secondMaxImg = np.zeros(img.shape)
        secondMaxImg.astype(np.uint8)
        for i in range(len(contours[secondMaxIdx])):
            y = contours[secondMaxIdx][i][0][0]
            x = contours[secondMaxIdx][i][0][1]
            secondMaxImg[x, y] = 255
    else:
        secondMaxImg = None
        secondMaxArea = 0
    
    return firstMaxImg, firstMaxArea, secondMaxImg, secondMaxArea
    
# to rectify the image
def rectifyImg(img, area):
    bTop, bBottom, bLeft, bRight = boundingRect(img)
    avgTop, avgBottom, avgLeft, avgRight = avgRect(img)
    
    topDis = avgTop - bTop
    bottomDis = bBottom - avgBottom
    leftDis = avgLeft - bLeft
    rightDis = bRight - avgRight
    
    avgHeight = avgBottom*1.0 - avgTop + 0.00001
    avgWidth = avgRight*1.0 - avgLeft + 0.00001
    
    return img, topDis/avgHeight, bottomDis/avgHeight, leftDis/avgWidth, rightDis/avgWidth
    
# compute mean excluding zero
def nonZeroMean(array):
    isZero = np.where(array==0, np.zeros(array.shape), np.ones(array.shape))
    if np.sum(isZero) == 0:
	    return 0;
    return np.sum(array)/np.sum(isZero)
    
def boundingRect(img):
    # which opencv draw contours will result in some noise
    if np.max(img) == 0:
        return 0, 0, 0, 0
    
    # get the max value of each column and each row in img
    eachRowMax = np.max(img, axis = 1)
    eachColMax = np.max(img, axis = 0)

    # determine 4 point: the highest, the lowest, the far left, the far right
    # the bounding box is drawn according (the highest, the far left), (the lowest, the far right)
    farLeft = np.argmax(eachColMax)
    farRight = eachColMax.shape[0] - np.argmax(eachColMax[::-1]) - 1
    highest = np.argmax(eachRowMax)
    lowest = eachRowMax.shape[0] - np.argmax(eachRowMax[::-1]) - 1

    return highest, lowest, farLeft, farRight
    
def avgRect(img):
    # which opencv draw contours will result in some noise
    if np.max(img) == 0:
        return 0, 0, 0, 0

    # compute the idx of each col max value
    top = np.argmax(img, axis = 0)
    left = np.argmax(img, axis = 1)
    imgRot = np.rot90(img, 2)
    bottom = np.argmax(imgRot, axis = 0)[::-1]
    right = np.argmax(imgRot, axis = 1)[::-1]

    # compute avg_rectangle
    topAvg = int(nonZeroMean(top))
    bottomAvg = int(nonZeroMean(bottom))
    leftAvg = int(nonZeroMean(left))
    rightAvg = int(nonZeroMean(right))
    bottomAvg = img.shape[0] - bottomAvg - 1
    rightAvg = img.shape[1] - rightAvg - 1
    return topAvg, bottomAvg, leftAvg, rightAvg
