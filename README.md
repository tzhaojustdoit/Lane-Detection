# Advanced Lane Finding

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Final Result"
[image2]: ./camera_cal/calibration2.jpg "Distorted Image Example"
[image3]: ./output_images/chestboard_undistorted.jpg "Undistorted Image Example"
[image4]: ./test_images/straight_lines1.jpg "Original Image"
[image5]: ./output_images/undist_straight_lines1.jpg "Undistorted Image"
[image51]: ./output_images/color_spaces.png "Color Space"
[image52]: ./output_images/threshold_fine_tune.png "Fine Tune"
[image6]: ./output_images/binary/hls_s.jpg "S channel of HLS"
[image7]: ./output_images/binary/hsv_v.jpg "V channel of HSV"
[image8]: ./output_images/binary/hls_s_th.jpg "Thresholded S channel"
[image9]: ./output_images/binary/hsv_v_th.jpg "Thresholded V channel"
[image10]: ./output_images/binary/final.jpg "Combined binary image"
[image11]: ./output_images/pre_warp.jpg "Undistorted image with source points drawn"
[image12]: ./output_images/warped.jpg "Warped image with destination points drawn"
[image13]: ./output_images/masked_test3.jpg "Binary image with windows drawn"
[image14]: ./output_images/fitted_test3.jpg "Binary image with fitted polynomials"
[image15]: ./output_images/final_test3.jpg "Final image"

[gif1]: ./gifs/project_video_output.gif "Project video output"
[gif2]: ./gifs/challenge_video_output.gif "Challenge video output"
[gif3]: ./gifs/project_video_diagnose.gif "Project video diagnose"
## Introduction
In this project, I used video streams from front-facing cameras on cars to identify the lane boundaries.

!["Project video output"][gif1]
## Camera Calibration
Camera calibration is done using the OpenCV Python library. The [Camera_Calibrator](./modules/camera_calibrator.py) class uses a series of camera images of a chestboard to obtain a set of object points and image points that are later used in image undistortion.
Image undistortion is done using the `cv2.calibrateCamera()` function.

Distorted Image\
!["Distorted Image Example"][image2]

Undistorted Image\
!["Undistorted Image Example"][image3]

## Pipeline
### 1. Undistortion

Original Image\
!["Original Image"][image4]

Undistorted Image\
!["Undistorted Image"][image5]

### 2. Binary Image
To more easily identify lane lines on the road, I used [gradient thresholding](./modules/thresholding.py) in different color spaces to convert to binary images.

First, I investigated into different color spaces. 

!["Color Spaces"][image51]

I found that the S channel of the HLS color space is able to consistently display lane lines closer to the ego vehicle clearly. However, it doesn't do a good job identifying lane lines further away. The V channel of the HSV color space is able to consistently identify lane lines further away. The B channel of the LAB color space is able to detect yellow lines very well. So I decided to use both and combine them together in the final binary image.

I used [ipywidgets](./thresholding_test.ipynb) to build a widget for parameters fine tuning.

!["Fine Tuning"][image52]

The S channel of HLS color space\
!["S channel of HLS"][image6]

The V channel of the HSV color space\
!["V channel of HSV"][image7]

For both channels, I applied thresholds on the color value, the magnitude of the gradient, and the direction of the gradient.

Thresholded S channel\
!["Thresholded S channel"][image8]

Thresholded V channel\
!["Thresholded V channel"][image9]

Finally, combine the two thresholded images to get the binary output. Lane lines look clear.

The combined binary image\
!["Combined binary image"][image10]

### 3. Perspective Transformation
The [PerspectiveTransformer](./modules/perspective_transformer.py) class uses source and destination points to calculate the transformation matrix M, which is later used in the `warp()` function. I hardcode the 4 source points and 4 destination points using a test image. I then used the same tranformation matrix M for all.

Undistorted image with source points drawn\
!["Undistorted image with source points drawn"][image11]

Warped image with destination points drawn\
!["Warped image with destination points drawn"][image12]

We used the same transform matrix M for every frame in the video. This is based on the assumption that the camera to road surface angle doesn't change. However, this is not true when the road is bumpy. More in the discussion session.

### 4. Region Masking
To find the lane lines using the binary image, I applied region masking to eliminate pixels that are clearly not lane lines.

First, I find histogram peaks in the horizontal direction. Then I used the two peaks as base lines to search around and then use sliding windows moving upward in the image to determine where the lane lines go. See `__get_inds_sliding_window` of [lane_finder.py](./modules/lane_finder.py)

Binary image with windows drawn\
!["Binary image with windows drawn"][image13]

In addition, since lane line positions don't tend to change much from frame to frame, I used polynomials from the previous frame to skip the sliding window. This speeds up the search. See `__get_inds_prior` of [lane_finder.py](./modules/lane_finder.py)

### 5. Fitting a Polynomial
To find the lane lines, I then fitted a second degree polynomial using the valid pixels I got from region masking. See `__get_line` of [lane_finder.py](./modules/lane_finder.py)

Binary image with fitted polynomials\
!["Binary image with fitted polynomials"][image14]
### 6. Calculating Curvatures and Offsets
Then I calculated the radius of curvature of the lane and the position of the vehicle with respect to center. See `__get_line` of [lane_finder.py](./modules/lane_finder.py)
### 6. Final Result
Here an example image of the result plotted back down onto the road in the distortion corrected image.\
!["Final image"][image15]

[Video output](./project_video_output.mp4)\
!["Project video output"][gif1]

This [video](./project_video_output_diagnose.mp4) shows which pixels are identified as lane lines.\
!["Project video diagnose"][gif3]
## Discussion
1. There are limitations to this traditional lane-finding approach. One is that it sometimes confuses other objects with lane lines.
For example, in the challenge video, it thinks the highway fence is the lane line. Maybe some smart thresholding using different color spaces would help ruling out other objects.
2. When one lane line completely dissapear from the camera view, the algorithm is unable to detect lane lines properly. This is due to using two peaks of a histogram to group left and right lane line pixels. Maybe thresholding on midpoint values would be able to detect if there is no obvious two peaks.
3. When bad detections (sudden change in curves) happen on consecutive frames, the algorithm would reset the search using the sliding window approach. The new detection could also be a bad one, and will take several more frames to reset again. I designed a novel way to solve this: confidence level: an integer value ranges from 0 to 5. 0 means low confidence in the lane line locations to be correct, 5 means high confidence in the lane line locations to be correct. When reset, the confidence level is set to 1. If a good detection happens, the confidence level increases by 1, but it doesn't go beyond 5. When a bad detection happens, the confidence level decreases by 1. If the confidence level drops to 0, reset and search using sliding window.
4. In the result video, I notice that the detected lane seems to spread out as the car moves over a bumpy surface. I think this is due to the changing in the camera angle. The transformation matrix used in the perspective transformation process is calculated using a straight lane camera image. The the matrix is used for all frames in the video. Obviously the matrix would be wrong when the camera angle changes. The curvature would also be wrong as a result. A dynamic transformation matrix would be needed to account for changing camera angle.
5. A question I thought about is should I do thresholding first or should I do perspective transform first? I think doing thresholding first would yield higher performance because thresholding converts a 256 bits per pixel, 3 color channels image to a 1 bit per pixel, 1 channel image. This would make perspective transformation much faster. However, I think doing perspective transformation first also has its benefit. When doing directional thresholding, a perspective transformed image would show much disirable lane directions, closer to vertical, thus allow us to get better direction thresholding results.

## Run
`python main.py`