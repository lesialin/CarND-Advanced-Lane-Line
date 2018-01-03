import matplotlib.pyplot as plt
import pickle
import numpy as np
from AdvLaneDet import  Threshold
from AdvLaneDet  import CameraCalibration as CamCalib
from moviepy.editor import VideoFileClip




#load obj/image points
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]

#set ROI
height =720
width =1280
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
ROI_vertices = np.array([[roi_upper_left,roi_down_left,triangle_lower_left,triangle_top,triangle_lower_right,roi_down_right,roi_upper_right]], dtype=np.int32)

clip = VideoFileClip("challenge_video.mp4")
for i in range(1,50,5):
	frame =  clip.get_frame(i)
	undist = CamCalib.calUndistort(frame, objpoints, imgpoints)

	#visualize
	mask_img = Threshold.region_of_interest(undist, ROI_vertices)
	plt.imshow(mask_img,cmap='gray')
	plt.show()