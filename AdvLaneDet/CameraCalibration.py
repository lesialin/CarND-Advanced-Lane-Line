import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
'''
This function is to get the objects 3D and images 2D coordinate for camera calibration.
Usage:
    input: 
    'camera_cal_dir' is the path of the dirctory with chessboard images
    'n_grid_x' and 'n_grid_y' are number of corner in chessboard x and y direction
    'draw' is True, will plot the chessboard corner results
    output:
    objpoints is the 3D coordinate of the object
    imgpoints is the 2D coordinate of the images
'''
def get_CameraCalib_Coord(camera_cal_dir,n_grid_x,n_grid_y,draw=False):
    # prepare object points 3-D coordinate, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    n_grid_x = 9
    n_grid_y = 6
    objp = np.zeros((n_grid_y*n_grid_x,3), np.float32)
    objp[:,:2] = np.mgrid[0:n_grid_x,0:n_grid_y].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    image_path = camera_cal_dir + '/*.jpg'
    images = glob.glob(image_path)
    if draw == True:
        n_col_plot = 2
        n_row_plot =int(len(images)/n_col_plot)
        fig = plt.figure(figsize= (6*n_col_plot,3*n_row_plot))
        plot_count = 0
        fig.subplots_adjust(hspace = 0.01, wspace = 0.05)
        print('Coner found in calibration chessboard:')
    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (n_grid_x,n_grid_y),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            if draw == True:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (n_grid_x,n_grid_y), corners, ret)
                ax = fig.add_subplot(n_row_plot, n_col_plot, plot_count+1)
                ax.axis('off')
                plot_count += 1
                plt.imshow(img)
    if draw == True:  
        plt.show()
        
    
    return objpoints,imgpoints
'''
This function is to undistort images by camera calibration mapping
Usage:
    input:
    'img' is the distorted image
    'objpoints and imgpoints' are the coordinate of objects and images respectively
    output:
    'undist' is undistorted image
'''
def calUndistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    if len(img.shape) > 2:
        undist = np.copy(img)
        for i in range(3):
            undist[:,:,i] = cv2.undistort(img[:,:,i], mtx, dist, None, mtx)    
    else:
        undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


    