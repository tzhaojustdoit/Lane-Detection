import cv2
import numpy as np
import logging
class PerspectiveTransformer:
    """ class that handles perspective transformation """
    def __init__(self, src, dst):
        """ src: source points
        dst: destination points """
        logging.info("[Perspective_Transformer] Initializing...")
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = np.linalg.inv(self.M)
        logging.info("[Perspective_Transformer] Done!")

    def warp(self, img):
        """
        Apply a perspective transform to rectify binary image ("birds-eye view")
        Returns the warped image
        """
        # Apply perpective transform
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, self.M, img_size, flags=cv2.INTER_NEAREST)
        return warped
    
    M = None
    Minv = None