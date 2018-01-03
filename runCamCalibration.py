from AdvLaneDet import CameraCalibration as CamCalib
import pickle
import cv2
import matplotlib.pyplot as plt
camera_calibraion_dir = 'camera_cal'
n_corner_x = 9
n_corner_y = 6
#get chessboard coordinates of objects and images for calibration 
objpoints,imgpoints = CamCalib.get_CameraCalib_Coord(camera_calibraion_dir,n_corner_x,n_corner_y,draw=False)
#dump to pickle file
dist_pickle = {}
dist_pickle["objpoints"] = objpoints
dist_pickle["imgpoints"] = imgpoints
with open('wide_dist_pickle.p', 'wb') as f:
    pickle.dump(dist_pickle, f)   
del objpoints, imgpoints
#load obj/image points
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]
# distortion correction by mapping objects and images coordinates
image_path = camera_calibraion_dir + '/calibration1.jpg'
img = cv2.imread(image_path)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


undist = CamCalib.calUndistort(gray, objpoints, imgpoints)
print('Dirstortion correction:')
fig = plt.figure(figsize= (6*2,3*1))
plt.subplot(121)
plt.imshow(gray,'gray')
plt.title('distorted image')
plt.axis('off')
plt.subplot(122)
plt.imshow(undist,'gray')
plt.title('undistorted image')
plt.axis('off')
plt.show()
