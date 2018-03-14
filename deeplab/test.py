import deeplab
import os
import sys
import basicprocess
import postprocess
import numpy as np
import cv2

for file in os.listdir("..\\Test"):
    deeplab.detectContainer("..\\Test\\" + file)