import numpy as np
import modules.util as util
import cv2

def get_binary(img):
    """ process the image into a binary so that the lane pixels are easy to see 
    returns the binary image """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = thresh(s_channel, 3, 15, 255, 34, 200, 0, np.pi/2.5)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = thresh(v_channel, 3, 180, 255, 20, 150, 0, np.pi/2.5)

    combined = np.zeros_like(s_binary)
    combined[((s_binary == 1) | (v_binary == 1))] = 1
    return combined

# def get_binary_test(img):
#     src_path = '/home/tzhao/src/SelfDrivingCarNanoDegree/Advanced-Lane-Finding/src' # src dir path
#     img = np.copy(img)
#     # Convert to HLS color space and separate the V channel
#     hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#     h_channel = hls[:,:,0]
#     l_channel = hls[:,:,1]
#     s_channel = hls[:,:,2]
#     util.write_binary_image(src_path + '/output_images/binary/hls_h.jpg',h_channel)
#     util.write_binary_image(src_path + '/output_images/binary/hls_l.jpg',l_channel)
#     util.write_binary_image(src_path + '/output_images/binary/hls_s.jpg',s_channel)
#     hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#     h_hsv_channel = hsv[:,:,0]
#     s_hsv_channel = hsv[:,:,1]
#     v_hsv_channel = hsv[:,:,2]
#     util.write_binary_image(src_path + '/output_images/binary/hsv_h.jpg',h_hsv_channel)
#     util.write_binary_image(src_path + '/output_images/binary/hsv_s.jpg',s_hsv_channel)
#     util.write_binary_image(src_path + '/output_images/binary/hsv_v.jpg',v_hsv_channel)
#     hls = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
#     b_channel = hls[:,:,0]
#     g_channel = hls[:,:,1]
#     r_channel = hls[:,:,2]
#     util.write_binary_image(src_path + '/output_images/binary/bgr_b.jpg',b_channel)
#     util.write_binary_image(src_path + '/output_images/binary/bgr_g.jpg',g_channel)
#     util.write_binary_image(src_path + '/output_images/binary/bgr_r.jpg',r_channel)

#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     util.write_binary_image(src_path + '/output_images/binary/gray.jpg',gray)

#     # hsv_v combined thresholding
#     hsv_v_abs_sobel_thresh = thresh(v_hsv_channel, 3, 180, 255, 20, 150, 0, np.pi/2.5)
#     util.write_binary_image(src_path + '/output_images/binary/hsv_v_th.jpg',hsv_v_abs_sobel_thresh)

#     # hls_s combined thresholding
#     hls_s_abs_sobel_thresh = thresh(s_channel, 3, 15, 255, 34, 200, 0, np.pi/2.5)
#     util.write_binary_image(src_path + '/output_images/binary/hls_s_th.jpg',hls_s_abs_sobel_thresh)


#     combined = np.zeros_like(hsv_v_abs_sobel_thresh)
#     combined[((hsv_v_abs_sobel_thresh == 1) | (hls_s_abs_sobel_thresh == 1))] = 1
#     util.write_binary_image(src_path + '/output_images/binary/final.jpg',combined)

#     return combined
    # # bgr_r thresholded
    # bgr_r_thres = [180, 255]
    # bgr_r_val = np.zeros_like(r_channel)
    # bgr_r_val[(r_channel >= bgr_r_thres[0]) & (r_channel <= bgr_r_thres[1])] = 1
    # util.write_binary_image(src_path + '/output_images/binary/bgr_r_val.jpg',bgr_r_val)

    # # hsv_v thresholded
    # hsv_v_thres = [180, 255]
    # hsv_v_val = np.zeros_like(v_hsv_channel)
    # hsv_v_val[(v_hsv_channel >= hsv_v_thres[0]) & (v_hsv_channel <= hsv_v_thres[1])] = 1
    # util.write_binary_image(src_path + '/output_images/binary/hsv_v_val.jpg',hsv_v_val)

    # # Sobel x on s channel
    # sobel_sx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    # abs_sobel_sx = np.absolute(sobel_sx) # Absolute x derivative to accentuate lines away from horizontal
    # scaled_sobel_sx = np.uint8(255*abs_sobel_sx/np.max(abs_sobel_sx))
    # sxbinary = np.zeros_like(scaled_sobel_sx)
    # sxbinary[(scaled_sobel_sx >= sx_thresh[0]) & (scaled_sobel_sx <= sx_thresh[1])] = 1
    # util.write_binary_image(src_path + '/output_images/binary/bin_test_sx_' + name_p + '.jpg',sxbinary)
    
    # # Sobel y on s channel
    # sobel_sy = cv2.Sobel(s_channel, cv2.CV_64F, 0, 1) # Take the derivative in x
    # abs_sobel_sy = np.absolute(sobel_sy) # Absolute x derivative to accentuate lines away from horizontal
    # scaled_sobel_sy = np.uint8(255*abs_sobel_sy/np.max(abs_sobel_sy))
    # sybinary = np.zeros_like(scaled_sobel_sy)
    # sybinary[(scaled_sobel_sx >= sx_thresh[0]) & (scaled_sobel_sx <= sx_thresh[1])] = 1
    # util.write_binary_image(src_path + '/output_images/bin_test_sx_' + name_p + '.jpg',sxbinary)

    # # Threshold color channel
    # s_binary = np.zeros_like(s_channel)
    # s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # # Stack each channel
    # binary = sxbinary | s_binary
    # return binary
    # return sxbinary

def thresh(img, ksize, val_low, val_high, mag_low, mag_high, dir_low, dir_high):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Calculate the gradient direction
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(sobelx)
    # thresholding with value, gradient magnitude, and gradient direction
    binary_output[(img >= val_low) & (img <= val_high) &
        (gradmag >= mag_low) & (gradmag <= mag_high) &
        (absgraddir >= dir_low) & (absgraddir <= dir_high)] = 1
    return binary_output