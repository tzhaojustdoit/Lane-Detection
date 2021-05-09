import modules.camera_calibrator as cc
import modules.perspective_transformer as pt
import modules.thresholding as th
import modules.lane_finder as lf
import modules.util as util

import glob
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import logging

src_path = os.getcwd() # src dir path
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize camera calibrator
filenames = glob.glob(src_path + '/camera_cal/calibration*.jpg')
calibrator = cc.Camera_Calibrator(filenames, 6, 9)

print("Testing undistort function on a chestboard image...")
chestboard_test = cv2.imread(src_path + '/camera_cal/calibration2.jpg')
print(chestboard_test.shape)
chestboard_undistorted = calibrator.undistort(chestboard_test)
util.plot_side_by_side([chestboard_test, chestboard_undistorted], ['Original Image', 'Undistorted Image'])