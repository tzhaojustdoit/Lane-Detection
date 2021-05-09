import numpy as np
import modules.line as line
import cv2
import matplotlib.pyplot as plt
import logging
class Lane_Finder:
    """ class that handles lane line finding """
    def __init__(self, img_shape):
        """ img_shape: image shape used by the lane finder. Lane_Finder assumes all camara images are of the same size """
        logging.info("[Lane_Finder] Initializing...")
        self.img_shape = img_shape
        logging.debug("[Lane_Finder] Image shape: {}.".format(self.img_shape))
        self.window_height = np.int(self.img_shape[0]//self.nwindows)
        self.ploty = np.linspace(0, self.img_shape[0]-1, self.img_shape[0])
        self.y_eval = np.max(self.ploty)
        self.left_line = line.Line() 
        self.right_line = line.Line()
        self.pixel_to_real = np.array([self.xm_per_pix / self.ym_per_pix ** 2, self.xm_per_pix / self.ym_per_pix, self.xm_per_pix])
        logging.info("[Lane_Finder] Done!")

    def process_frame(self, binary_warped):
        """ process a frame """
        ## for debugging
        ## should be commented out
        # self.binary_masked = []
        
        nonzero = binary_warped.nonzero() # none zero pixels
        nonzeroy = np.array(nonzero[0]) # x
        nonzerox = np.array(nonzero[1]) # y
        histogram = [] # for grouping left and right lane line pixels
        self.__get_line(self.left_line, 'left', binary_warped, nonzerox, nonzeroy, histogram)
        self.__get_line(self.right_line, 'right', binary_warped, nonzerox, nonzeroy, histogram)

    def __get_line(self, line, line_name, binary_warped, nonzerox, nonzeroy, histogram):
        """ get the lane line
        line: a Line object
        line_name: left or right
         """
        line_fit = []
        line_inds = []
        logging.debug("[Lane_Finder] Confidence level {}: {}".format(line_name, line.confidence_level))
        if line.confidence_level <= 0 or len(line.curr_fit) == 0: # confidence level is too low, search using sliding window
            logging.debug("[Lane_Finder] {} reset.".format(line_name))
            if histogram == []:
                histogram = np.sum(binary_warped[self.img_shape[0]//2:,:], axis=0)
            midpoint = np.int(histogram.shape[0]//2)
            if (line_name == 'left'):
                line_inds = self.__get_inds_sliding_window(np.argmax(histogram[:midpoint]), nonzerox, nonzeroy)
            else:
                line_inds = self.__get_inds_sliding_window(np.argmax(histogram[midpoint:]) + midpoint, nonzerox, nonzeroy)
            line_x = nonzerox[line_inds]
            line_y = nonzeroy[line_inds] 
            # visualize binary masked image for debugging
            # should be commented
            # self.__visualize(binary_warped, line_x, line_y, line_name, True)
            if len(line_x) == 0:
                return
            line_fit = np.polyfit(line_y, line_x, 2)
            line.confidence_level = 1 # set confidence level to 1
            line.add_fit(line_fit)
        else: # confidence level is above 0, search using previous parabola
            logging.debug("[Lane_Finder] Search using prior info.")
            prior_fit = line.curr_fit
            line_inds = self.__get_inds_prior(prior_fit, nonzerox, nonzeroy)
            line_x = nonzerox[line_inds]
            line_y = nonzeroy[line_inds] 
            # visualize binary masked image for debugging
            # should be commented
            # self.__visualize(binary_warped, line_x, line_y, line_name, False)
            if len(line_x) == 0:
                line.confidence_level -= 1 # decrese confidence level by 1
                return
            line_fit = np.polyfit(line_y, line_x, 2)
            if len(line_fit) != 0:
                if self.__is_valid(line_fit, line.curr_fit): # check if the detection is valid or not
                    logging.debug("[Lane_Finder] Good detection.")
                    if line.confidence_level < 5:
                        line.confidence_level += 1 # increase confidence level, not above 5
                    line.add_fit(line_fit)                
                else:  # bad detection
                    logging.debug("[Lane_Finder] Bad detection.")
                    line.confidence_level -= 1 # decrease confidence level
                    line.add_fit(line.curr_fit)

        logging.debug("[Lane_Finder] Average fit: {}".format(line.average_fit))
        # get x values for plotting
        line.best_fitx = line.average_fit[0]*self.ploty**2 + line.average_fit[1]*self.ploty + line.average_fit[2]
        # get curvature
        fit_real = line_fit * self.pixel_to_real
        radius_of_curvature = ((1 + (2*fit_real[0]*self.y_eval*self.ym_per_pix + fit_real[1])**2)**1.5) / np.absolute(2*fit_real[0])
        logging.debug("[Lane_Finder] Curvature: {}".format(radius_of_curvature))
        # limit the update frequency to once every 7 frames.
        if line.radius_of_curvature == None:
            line.radius_of_curvature = radius_of_curvature
        elif len(line.cumulated_rad_of_curvs) <= 7:
            line.cumulated_rad_of_curvs.append(radius_of_curvature)
        else:
            line.radius_of_curvature = np.mean(line.cumulated_rad_of_curvs)
            line.cumulated_rad_of_curvs.clear()
        # get offset
        line_base_pos = np.abs(line.best_fitx[-1] - self.img_shape[1] // 2) * self.xm_per_pix     
        logging.debug("[Lane_Finder] Distance: {}".format(line_base_pos))   
        # limit the update frequency to once every 7 frames.
        if line.line_base_pos == None:
            line.line_base_pos = line_base_pos
        elif len(line.cumulated_line_base_pos) <= 7:
            line.cumulated_line_base_pos.append(line_base_pos)
        else:
            line.line_base_pos = np.mean(line.cumulated_line_base_pos)
            line.cumulated_line_base_pos.clear()        

    def __get_inds_sliding_window(self, base, nonzerox, nonzeroy):
        """ get lane pixel indexes using sliding window method """
        # Create empty lists to receive lane pixel indices
        lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.img_shape[0] - (window+1)*self.window_height
            win_y_high = self.img_shape[0] - window*self.window_height
            win_x_low = base - self.margin_sliding_window
            win_x_high = base + self.margin_sliding_window
        
            # Identify the nonzero pixels in x and y within the window #
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]
        
            # Append these indices to the lists
            lane_inds.append(good_inds)
        
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > self.minpix:
                base = np.int(np.mean(nonzerox[good_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            lane_inds = np.concatenate(lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        return lane_inds

    def __get_inds_prior(self, prior_fit, nonzerox, nonzeroy):
        """ get lane pixel indexes using prior fitted polynomial """
        lane_inds = ((nonzerox > (prior_fit[0]*(nonzeroy**2) + prior_fit[1]*nonzeroy + 
                            prior_fit[2] - self.margin_polynomial)) & (nonzerox < (prior_fit[0]*(nonzeroy**2) + 
                            prior_fit[1]*nonzeroy + prior_fit[2] + self.margin_polynomial)))
        return lane_inds

    def __is_valid(self, curr, prev):
        """ check if the detection is valid """
        diffs = np.abs(curr - prev) # calculate the coeff difference
        logging.debug("[Lane_Finder] Coeff diff: {}".format(diffs))
        if diffs[0] < self.diff_thresh[0] and diffs[1] < self.diff_thresh[1] and diffs[2] < self.diff_thresh[2]:
            return True
        return False

    def __visualize(self, binary_warped, line_x, line_y, line_name, is_window):
        """ visualize the binary masked image for debugging """
        # Create an image to draw on and an image to show the selection window
        if len(self.binary_masked) == 0 :
            self.binary_masked = np.dstack((binary_warped, binary_warped, binary_warped))*255
        if line_name == 'left':
            if is_window:
                self.binary_masked[line_y, line_x] = [255, 0, 0]
            else:
                self.binary_masked[line_y, line_x] = [0, 255, 0]
        else:
            if is_window:
                self.binary_masked[line_y, line_x] = [0, 0, 255]
            else:
                self.binary_masked[line_y, line_x] = [0, 255, 0]

    left_line = line.Line() # left history
    right_line = line.Line() # right history
    nwindows = 7 # number of windows
    window_height = None # height of windows - based on nwindows above and image shape
    margin_sliding_window = 70 # width of the windows +/- margin
    margin_polynomial = 35 # width of the polynomial +/- margin
    minpix = 50 # minimum number of pixels found to recenter window
    img_shape = None # image shape produced by the camera
    xm_per_pix = 3.7/700 # meters per pixil in x dir
    ym_per_pix = 30/720  # meters per pixil in y dir
    pixel_to_real = None  # scale by which each coefficient term need to be scaled to convert from pixel space to real
    diff_thresh = [0.0001, 0.1, 100]
    ploty = [] # y values for plotting
    y_eval = None # y value of the image bottom, evaluate curvature with it
    binary_masked = [] # binary masked image for debugging