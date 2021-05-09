import numpy as np
from statistics import mean
class Line():
    """ class that holds lane line info """
    def __init__(self):
        # inceases when a detection is good, decreases when a detection is bad
        # range is between 0 to 5, 0 means no confidence in the current dectection, should reset; 5 means high confidence
        self.confidence_level = 0
        # polynomial coefficients of the current fit of the line
        self.curr_fit = []
        # weighted average of the recent fits.
        self.average_fit = []
        # x values calculated using coefficients averaged over the last n iterations
        self.best_fitx = []
        # radius of curvatures of frames in the current window. When several frames are cummulated, caculate average.
        self.cumulated_rad_of_curvs = []
        # radius of curvature of the line in meters, averaged over the last several frames
        self.radius_of_curvature = None 
        # distances in the current window. When several frames are cummulated, caculate average.
        self.cumulated_line_base_pos = []        
        # distance in meters of vehicle center from the line, average over the last several frames
        self.line_base_pos = None 

    def add_fit(self, fit):
        """ Add a new fit coeffs, compute average fit coeffs, exponential smoothing"""
        if self.curr_fit == []:
            self.curr_fit = fit
            self.average_fit = fit
            return
        self.average_fit = fit * self.coef + self.curr_fit * (1 - self.coef)
        self.curr_fit = fit

    coef = 0.6 # exponential smoothing coefficent