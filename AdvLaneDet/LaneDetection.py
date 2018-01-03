import numpy as np
import cv2
import matplotlib.pyplot as plt
from AdvLaneDet import Threshold
import pickle
from AdvLaneDet import  CameraCalibration as CamCalib

# Define a class to receive the characteristics of each line detection
class LaneDetection():
    def __init__(self):
        #polynomial coefficients for the most recent fit
        self.left_fit = None
        self.right_fit  = None
        #radius of curvature of the line in some units
        self.left_curvature = None 
        self.right_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #the maximum distance(in meter) away from center
        self.max_bais_center = 0.3
        #Camera calibration
        self.CamCalib_objpoints = None
        self.CamCalib_imgpoints = None
        #the ROI region of the lane
        self.ROI_vertices = None
        #frame size = (width,height)
        self.img_size = None
        #binary threshold
        self.ksize = None # Choose a larger odd number to smooth gradient measurements
        self.grad_thrshold = None
        self.s_threhold = None
        self.mag_thresh = None
        #perspective transform
        self.M = None
        self.Minv = None
        #x and y direction meter/pixel
        self.ym_per_pix = None
        self.xm_per_pix = None
        #number of detected pts in lane
        self.n_detected_pts_left = None
        self.n_detected_pts_right = None
        self.n_prev_detected_pts_left = None
        self.n_prev_detected_pts_right = None

    def getLanePloy(self,binary_warped,visualize=False):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[np.int(binary_warped.shape[0]/2):,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        #number of detected points in the lane
        self.n_prev_detected_pts_left = self.n_detected_pts_left
        self.n_prev_detected_pts_right = self.n_detected_pts_right
        self.n_detected_pts_left = len(leftx)
        self.n_detected_pts_right = len(rightx)

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        if visualize == True:
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            plt.imshow(out_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.show()
        self.left_fit = left_fit
        self.right_fit = right_fit
        

    def getNextLanePoly(self,binary_warped,left_fit,right_fit,visualize=False):
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
        left_fit[1]*nonzeroy + left_fit[2] + margin))) 

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
        right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        self.left_fit = left_fit
        self.right_fit = right_fit
        #number of detected points in the lane
        self.n_prev_detected_pts_left = self.n_detected_pts_left
        self.n_prev_detected_pts_right = self.n_detected_pts_right
        self.n_detected_pts_left = len(leftx)
        self.n_detected_pts_right = len(rightx)
        

        if visualize == True:
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                          ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                          ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
            plt.imshow(result)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.show()

            
        

    def warper(self,img, src, dst):
        # Compute and apply perpective transform
        img_size = (img.shape[1], img.shape[0])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
        return warped

    def drawLines(self,img,verices, color=[255, 0, 0], thickness=1):
        left_x1 = verices[0][0]
        left_y1 = verices[0][1]
        left_x2 = verices[1][0]
        left_y2 = verices[1][1]
        cv2.line(img, (left_x1, left_y1), (left_x2, left_y2), color, thickness)
        right_x1 = verices[2][0]
        right_y1 = verices[2][1]
        right_x2 = verices[3][0]
        right_y2 = verices[3][1]
        cv2.line(img, (right_x1, right_y1), (right_x2, right_y2), color, thickness)
        upper_right_x1 =  right_x1
        upper_right_y1 = right_y1
        upper_left_x1 = left_x2
        upper_left_y1 = left_y2
        cv2.line(img, (upper_right_x1, upper_right_y1), (upper_left_x1, upper_left_y1), color, thickness)
        lower_right_x1 =  right_x2
        lower_right_y1 = right_y2
        lower_left_x1 = left_x1
        lower_left_y1 = left_y1
        cv2.line(img, (lower_right_x1, lower_right_y1), (lower_left_x1, lower_left_y1), color, thickness)

    def calCurvature(self,left_fit,right_fit,img_size,xm_per_pix,ym_per_pix):
        
        y_eval = img_size[1]
        ploty = np.linspace(0, img_size[1]-1, img_size[1])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        self.left_curvature = left_curverad
        self.right_curvature= right_curverad


    def drawLaneRegion(self,binary_warped,undist,Minv,left_fit,right_fit):
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        left_lane_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_lane_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        lane_pts = np.hstack((left_lane_pts, right_lane_pts))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([lane_pts]), (0,255, 0))
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        return result

    def Postion_in_Lane(self,left_fit,right_fit,img_size):
        y_eval = img_size[1]
        left_fitx = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
        right_fitx = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
        position = (left_fitx + right_fitx)/2
        return position
        
'''
This function is to detect lane area  in video frames.
Usage:
    input:
     'image' is the frame in a video stream
     'LD' is a class object of LaneDetection
    output:
        'colored_lane' is the frame with detected colored land area
'''
def LaneDetection_pipline(image,LD,vid=1,visualize=False):
    
    if LD.img_size is None:
        LD.img_size = (image.shape[1],image.shape[0])
    
    if (LD.CamCalib_objpoints is None) or (LD.CamCalib_imgpoints is None):
        #load obj/image points
        dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
        LD.objpoints = dist_pickle["objpoints"]
        LD.imgpoints = dist_pickle["imgpoints"]
    
    if (LD.ksize is None) or (LD.grad_thrshold is None) or (LD.s_threhold is None) or (LD.mag_thresh is None):
        LD.ksize = 15 # Choose a larger odd number to smooth gradient measurements
        LD.grad_thrshold = (40,100)
        LD.s_threhold = (170,255)
        LD.mag_thresh = (130,255)

    if LD.ROI_vertices is None:
   
        #set ROI
        if vid  == 1:
            height = LD.img_size[1]
            width = LD.img_size[0]
            shirk_roi_upper_edge = np.round(width *0.48)
            roi_upper = np.round(0.6*height)
            roi_lower = np.round(height*0.93)
            roi_upper_left = (shirk_roi_upper_edge,roi_upper)
            roi_upper_right = (width-shirk_roi_upper_edge, roi_upper)
            roi_down_left = (np.round(width*0.1), roi_lower)
            roi_down_right = (np.round(width-(width*0.1)),roi_lower)
            # mask ROI insize triangle
            triangle_lower_left = (np.round(width*0.28),roi_lower)
            triangle_lower_right = (np.round(width-(width*0.28)),roi_lower)
            triangle_top = (np.round(width*0.5),np.round(0.65*height))
            #  verices of ROI
            LD.ROI_vertices = np.array([[roi_upper_left,roi_down_left,triangle_lower_left,triangle_top,triangle_lower_right,roi_down_right,roi_upper_right]], dtype=np.int32)
    
        if vid == 2:
            height = LD.img_size[1]
            width = LD.img_size[0]
            shirk_roi_upper_edge = np.round(width *0.42)
            roi_upper = np.round(0.68*height)
            roi_lower = np.round(height*0.96)
            roi_upper_left = (shirk_roi_upper_edge,roi_upper)
            roi_upper_right = (width-shirk_roi_upper_edge, roi_upper)
            roi_down_left = (np.round(width*0.16), roi_lower)
            roi_down_right = (np.round(width-(width*0.08)),roi_lower)
            # mask ROI insize triangle
            triangle_lower_left = (np.round(width*0.28),roi_lower)
            triangle_lower_right = (np.round(width-(width*0.26)),roi_lower)
            triangle_top = (np.round(width*0.5),np.round(0.70*height))
            LD.ROI_vertices = np.array([[roi_upper_left,roi_down_left,triangle_lower_left,triangle_top,triangle_lower_right,roi_down_right,roi_upper_right]], dtype=np.int32)
    if (LD.M is None) or (LD.Minv is None):
        #src and dist coordinate for perception transformation
        src = np.float32([[ 585,  460],
                          [ 203,  684],
                          [1126,  684],
                          [ 695,  460]])
        dst = np.float32([[ 213,    0],
                          [ 213,  720],
                          [1066,  720],
                          [1066,    0]])
        LD.M = cv2.getPerspectiveTransform(src, dst)
        LD.Minv = cv2.getPerspectiveTransform(dst, src)
    if (LD.ym_per_pix is None) or (LD.xm_per_pix is None):
        # Define conversions in x and y from pixels space to meters
        LD.ym_per_pix = 30/720 # meters per pixel in y dimension
        LD.xm_per_pix = 3.7/700 # meters per pixel in x dimension

    undist = CamCalib.calUndistort(image, LD.objpoints, LD.imgpoints)
    binary = Threshold.pipeline(undist, LD.objpoints, LD.imgpoints, LD.ROI_vertices,ksize=LD.ksize, grad_thresh=LD.grad_thrshold, s_thresh=LD.s_threhold,mag_thresh=LD.mag_thresh, debug=False)
    binary_warped = cv2.warpPerspective(binary, LD.M, LD.img_size, flags=cv2.INTER_NEAREST)
    
    if (LD.n_prev_detected_pts_left is None) or (LD.n_prev_detected_pts_right is None):
        LD.getLanePloy(binary_warped,visualize=visualize)
    elif  (LD.n_detected_pts_left < LD.n_prev_detected_pts_left/2) or (LD.n_detected_pts_right < LD.n_prev_detected_pts_right/2):
        LD.getLanePloy(binary_warped,visualize=visualize)
    elif LD.line_base_pos > LD.max_bais_center:
        LD.getLanePloy(binary_warped,visualize=visualize)
    else:
        LD.getNextLanePoly(binary_warped,LD.left_fit,LD.right_fit,visualize=visualize)
    
    LD.calCurvature(LD.left_fit,LD.right_fit,LD.img_size,LD.xm_per_pix,LD.ym_per_pix)
    colored_lane = LD.drawLaneRegion(binary_warped,undist,LD.Minv,LD.left_fit,LD.right_fit)
    position = LD.Postion_in_Lane(LD.left_fit,LD.right_fit,LD.img_size)
    LD.line_base_pos = (LD.img_size[0]/2 - position)*LD.xm_per_pix
    if LD.line_base_pos  > 0 :
        text = 'right'
    else:
        text = 'left'
    cv2.putText(colored_lane,'Radius of Curvature: %.2fm' % LD.left_curvature,(20,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(colored_lane,'Away Center: %.2fm %s' % (abs(LD.line_base_pos),text),(20,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

    return colored_lane

def LD_pipeline_VidProc(clip, LD,vid,visualize):
    def LD_pipeline(image):
        return LaneDetection_pipline(image, LD,vid,visualize)
    return clip.fl_image(LD_pipeline)


    