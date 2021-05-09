import matplotlib.pyplot as plt
import numpy as np
import cv2
def plot_side_by_side(images, names):
    """ plot two images side by side """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(images[0])
    ax1.set_title(names[0], fontsize=25)
    ax2.imshow(images[1])
    ax2.set_title(names[1], fontsize=25)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def draw_lines(img, lines, color=[225, 0, 0], thickness=5):
    """
    Draw lines on the image
    """
    for x1,y1,x2,y2 in lines:
        cv2.line(img, (x1, y1), (x2, y2), color, thickness) 

def draw_lanes(warped, undist, ploty, left_fitx, right_fitx, y_eval, xm_per_pix, ym_per_pix, Minv):
    """ draw lanes on the undistorted image """
    try:
    # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        return result
    except:
        return undist

def put_text(image, left_cr, left_dist, right_cr, right_dist):
    """ put curvature and distance text on an image """
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # org 
    # org1 = (100, 250) 
    # org2 = (100, 300)
    # org3 = (700, 250)
    # org4 = (700, 300)
    org5 = (100, 100)
    org6 = (100, 150)
    # fontScale 
    fontScale = 2
    # Blue color in BGR 
    color = (0, 0, 255) 
    # Line thickness of 2 px 
    thickness = 2  
    # cv2.putText(image, "left curv: {:.2f}".format(left_cr), org1, font,  
    #                fontScale, color, thickness, cv2.LINE_AA) 
    # cv2.putText(image, "left dist: {:.2f}".format(left_dist), org2, font,  
    #                fontScale, color, thickness, cv2.LINE_AA) 
    # cv2.putText(image, "right curv: {:.2f}".format(right_cr), org3, font,  
    #                fontScale, color, thickness, cv2.LINE_AA) 
    # cv2.putText(image, "right dist: {:.2f}".format(right_dist), org4, font,  
    #                fontScale, color, thickness, cv2.LINE_AA) 

    offset = (left_dist - right_dist) / 2
    avg_curv = np.mean([left_cr, right_cr])
    cv2.putText(image, "Average curvature: {:10.2f}m".format(avg_curv), org5, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
    cv2.putText(image, "Center offset:      {:10.2f}m".format(offset), org6, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 

def write_binary_image(fn, binary):
    """ write binary image to a file """
    binary_out = np.dstack((binary, binary, binary)) * 255
    cv2.imwrite(fn,binary_out)