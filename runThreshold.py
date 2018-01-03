import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from AdvLaneDet import  Threshold
from AdvLaneDet import  CameraCalibration as CamCalib
import pickle

#load obj/image points
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]
#set ROI
height =720
width =1280
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
ROI_vertices = np.array([[roi_upper_left,roi_down_left,triangle_lower_left,triangle_top,triangle_lower_right,roi_down_right,roi_upper_right]], dtype=np.int32)
#set threshold 
ksize = 15 
grad_thrshold = (40,100)
s_threhold = (170,255)
mag_thresh = (130,255)

images = glob.glob('test_images/*.jpg')
for fname in images:
    image = plt.imread(fname)
    #distortion correction
    undist = CamCalib.calUndistort(image, objpoints, imgpoints)
    binary = Threshold.pipeline(undist, objpoints, imgpoints, ROI_vertices,ksize=ksize, grad_thresh=grad_thrshold, s_thresh=s_threhold,mag_thresh=mag_thresh, debug=True)