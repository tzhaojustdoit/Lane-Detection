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

# test undistort, thresholding, warp, sliding_window mask, polyfit, and final image on the test images
test_image_file_names = os.listdir(src_path + "/test_images/")
for test_image_file_name in test_image_file_names:
    test_image = mpimg.imread(os.path.join(src_path + '/test_images/',test_image_file_name))
    undistorted = calibrator.undistort(test_image)
    undist_name = os.path.join(src_path + "/output_images", 'undist_' + test_image_file_name)
    print("writing to: {}...".format(undist_name))
    cv2.imwrite(undist_name, cv2.cvtColor(undistorted, cv2.COLOR_RGB2BGR))

    binary = th.get_binary(undistorted)
    bin_name = os.path.join(src_path + "/output_images", 'bin_' + test_image_file_name)
    print("writing to: {}...".format(bin_name))
    cv2.imwrite(bin_name, binary * 255)

    warped = persp_transformer.warp(binary)
    warped_name = os.path.join(src_path + "/output_images", 'warped_' + test_image_file_name)
    print("writing to: {}...".format(warped_name))
    cv2.imwrite(warped_name, warped * 255)

    # initialize lane_finder
    lane_finder = lf.Lane_Finder(warped.shape)
    # process a frame
    lane_finder.process_frame(warped)

    print("curvature: {}, {}".format(lane_finder.left_line.radius_of_curvature, lane_finder.right_line.radius_of_curvature))

    masked_out = lane_finder.out_boxed
    masked_name = os.path.join(src_path + "/output_images", 'masked_' + test_image_file_name)
    print("writing to: {}...".format(masked_name))
    cv2.imwrite(masked_name, masked_out)

    plt.close()
    plt.imshow(warped)
    plt.plot(lane_finder.left_line.best_fitx, lane_finder.ploty, color='white')
    plt.plot(lane_finder.right_line.best_fitx, lane_finder.ploty, color='white')
    fitted_name = os.path.join(src_path + "/output_images", 'fitted_' + test_image_file_name)
    print("writing to: {}...".format(fitted_name))
    plt.savefig(fitted_name)

    final_out = util.draw_lanes(warped, undistorted, lane_finder.ploty, lane_finder.left_line.best_fitx, lane_finder.right_line.best_fitx, lane_finder.y_eval, lane_finder.xm_per_pix, lane_finder.ym_per_pix, persp_transformer.Minv)
    final_name = src_path + "/output_images/final_" + test_image_file_name
    print("writing to: {}...".format(final_name))
    cv2.imwrite(final_name, cv2.cvtColor(final_out, cv2.COLOR_RGB2BGR))