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

# Initialize perspective transformer
# Loading a sample image
sample = mpimg.imread(src_path + '/test_images/straight_lines1.jpg')
sample_undistorted = calibrator.undistort(sample)
img_size = sample_undistorted.shape
src = np.float32(
 [[(img_size[1] / 2) - 60, img_size[0] / 2 + 99],
 [((img_size[1] / 6) - 10), img_size[0]],
 [(img_size[1] * 5 / 6) + 60, img_size[0]],
 [(img_size[1] / 2 + 63), img_size[0] / 2 + 100]])
dst = np.float32(
 [[(img_size[1] / 4), 0],
 [(img_size[1] / 4), img_size[0]],
 [(img_size[1] * 3 / 4), img_size[0]],
 [(img_size[1] * 3 / 4), 0]])
persp_transformer = pt.PerspectiveTransformer(src, dst)

#%% Optional to run
print("Testing perspective transformation on a staight line image")
pre_warp = np.copy(sample_undistorted)
src_pts = src.astype(int)
src_lines = [
    np.append(src_pts[0], src_pts[1]),
    np.append(src_pts[1], src_pts[2]),
    np.append(src_pts[2], src_pts[3]),
    np.append(src_pts[3], src_pts[0])
]
util.draw_lines(pre_warp, src_lines)
warped = persp_transformer.warp(sample_undistorted)
dst_pts = dst.astype(int)
dst_lines = [
    np.append(dst_pts[0], dst_pts[1]),
    np.append(dst_pts[1], dst_pts[2]),
    np.append(dst_pts[2], dst_pts[3]),
    np.append(dst_pts[3], dst_pts[0])
]
util.draw_lines(warped, dst_lines)

util.plot_side_by_side([pre_warp, warped], ['Undistorted Image with Source Points Drawn', 'Warped Result with Dest. Points Drawn'])
cv2.imwrite(src_path + '/output_images/pre_warp.jpg', cv2.cvtColor(pre_warp, cv2.COLOR_RGB2BGR))
cv2.imwrite(src_path + '/output_images/warped.jpg', cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))