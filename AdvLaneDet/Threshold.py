import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from AdvLaneDet import  CameraCalibration as CamCalib
import pickle
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    #gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gray = img[:,:,0]
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel))
    
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])]= 1
    # 6) Return this mask as your binary_output image
   
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = np.absolute(cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=sobel_kernel))
    sobely = np.absolute(cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=sobel_kernel))
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    dir_grad = np.arctan2(sobely,sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output= np.zeros_like(gray)
    # 6) Return this mask as your binary_output image
    binary_output[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1
    
    return binary_output


def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    s_channel = hls[:,:,2]
    # 3) Return a binary image of threshold result
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

# Edit this function to create your own pipeline.
def pipeline(img, objpoints, imgpoints, ROI_vertices,ksize=3, grad_thresh=(170, 255), s_thresh=(20, 100), mag_thresh=(170,255), debug=False):
    
    #gradient binary
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=grad_thresh)
    #saturation binary
    s_binary = hls_select(img, thresh=s_thresh)
    #direction/magnitude binary
    mag_binary = mag_threshold(img, sobel_kernel=ksize, mag_thresh=mag_thresh)
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(np.pi/3, np.pi/2))
    
    combined = np.zeros_like(dir_binary)
    combined[((mag_binary == 1) & (dir_binary == 1)) | (gradx == 1) | (s_binary == 1)] = 1
    binary = region_of_interest(combined, ROI_vertices)

    if debug==True:
        #mask image
        mask_img = region_of_interest(img, ROI_vertices)
        plt.imshow(img)
        plt.title('origin')
        plt.show()
        plt.imshow(mask_img)
        plt.title('ROI mask image')
        plt.show()
        mag_dir_combined = np.zeros_like(dir_binary)
        mag_dir_combined[((mag_binary == 1) & (dir_binary == 1))] = 1
        plt.imshow(mag_dir_combined,cmap = 'gray')
        plt.title('mag and direction binary')
        plt.show()
        color_binary = np.dstack(( gradx, s_binary, mag_dir_combined)) * 255
        plt.imshow(color_binary)
        plt.title('Threshold image: gradx(R)/saturation(G)/direction(B)')
        plt.show()
        plt.imshow(binary,cmap ='gray')
        plt.title('Binary image')
        plt.show()

    return binary
    

 