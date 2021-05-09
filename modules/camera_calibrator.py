import numpy as np
import cv2
import logging
class Camera_Calibrator:
    """ class that handles camera calibration """
    def __init__(self, filenames, nx, ny):  
        """ 
        filenames: chestboard image file names used for camera calibration
        nx: number of vertices in x dir
        ny: number of vertices in y dir
         """
        logging.info("[Camera_Calibrator] Initializing...")    
        objp = np.zeros((ny*nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:ny,0:nx].T.reshape(-1,2)
        # Step through the list and search for chessboard corners
        img = None
        for fname in filenames:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
        logging.debug("[Camera_calibrator] Obtained {} sets of obj and img points".format(len(self.objpoints)))
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, img.shape[1:], None, None)
        logging.info("[Camera_Calibrator] Done!")   

    def undistort(self, img):
        """
        Apply a distortion correction to the image
        Returns the undistorted image
        """      
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undist

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    mtx = None # camera matrix
    dist = None # distance coefficients
