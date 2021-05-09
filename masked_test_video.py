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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize camera calibrator
filenames = glob.glob(src_path + '/camera_cal/calibration*.jpg')
calibrator = cc.Camera_Calibrator(filenames, 6, 9)

# Loading a sample image
sample = mpimg.imread(src_path + '/test_images/straight_lines1.jpg')
sample_undistorted = calibrator.undistort(sample)
img_size = sample_undistorted.shape

# Initialize perspective transformer
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

# Pipeline
def process_image(image):
    """ process the image, find lane lines
    returns an image of lane area plotted back down onto the road. """
    undist = calibrator.undistort(image) # undistort
    binary = th.get_binary(undist) # thresholding
    warped = persp_transformer.warp(binary) # perspective transform
    lane_finder.process_frame(warped) # find lane line
    final_out = lane_finder.binary_masked # get binary masked image
    return final_out

# Generate video
lane_finder = lf.Lane_Finder(sample.shape)
project_clip = VideoFileClip(src_path + '/project_video.mp4')
project_clip_processed = project_clip.fl_image(process_image)
project_clip_processed.write_videofile(src_path + '/project_video_output_diagnose.mp4', audio=False)