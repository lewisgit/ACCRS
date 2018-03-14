# some basic function used to process image

import numpy as np
import cv2
import os
import sys
from scipy import misc
from scipy.io import loadmat as sio
import PIL.Image as Image
import struct

# resize img(large -> small) and generate .ppm image for densecrf
def imgPretreat(orgImg):
    orgImgShape = np.shape(orgImg)
    ratio = 500.0/np.max(orgImgShape)
    newImg = cv2.resize(orgImg, (int(orgImgShape[1]*ratio), int(orgImgShape[0]*ratio)))
    return newImg


# change .mat generared by CNN to .png
def mat2png(matPath):
    # generate a colormap
    palette=[]
    for i in range(256):
        palette.extend((i,i,i))
    palette[:3*2]=np.array([[0, 0, 0],
                            [0, 128, 0]], dtype='uint8').flatten()
    
    # change file type
    matFile = sio(matPath)
    matFile = matFile['data']
    labelImg = np.argmax(matFile,axis=2).astype(np.uint8)
    labelImg = cv2.flip(labelImg, 1)    # left to right flip
    labelImg = np.rot90(labelImg, 1)
    return labelImg*255
	
# change .bin generared by densecrf to .png
def bin2png(binPathName):
    binName = os.path.basename(binPathName)
    binFile = open(binPathName, 'rb')
    data = binFile.read()
    binFile.close()
	
	# read size
    dataByte = bytearray(data)
    row = struct.unpack('i', dataByte[0:4])[0]
    col = struct.unpack('i', dataByte[4:8])[0]
    channel = struct.unpack('i', dataByte[8:12])[0]  # channel is always 1

    # read data
    numel = row * col * channel
    map = np.array(dataByte[12:], dtype=np.int8)
    map.dtype = 'int16'
    map = map.astype(np.uint8)

    # save label as gray image
    mapGray = np.reshape(map, (row, col), order='F')
    mapGray = mapGray * 255
    return mapGray
 
# generate the masked scene image
def mask(orgImg, labelImg):
    labelImg.astype(np.uint8)
    if len(labelImg.shape) == 2: # if gray img, generate color img
        numel = labelImg.shape[0]*labelImg.shape[1]
        colorArray = np.reshape(labelImg, numel, 'F')
        colorArray = np.append(np.zeros(numel, dtype=np.uint8), colorArray)
        colorArray = np.append(colorArray, np.zeros(numel, dtype=np.uint8))
        labelImg = np.reshape(colorArray, (labelImg.shape[0], labelImg.shape[1], 3), 'F')
    maskImg = np.where(labelImg==0, orgImg, 0.7*orgImg+0.3*labelImg)
    return maskImg
