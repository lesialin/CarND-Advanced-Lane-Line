from moviepy.editor import VideoFileClip
from AdvLaneDet import LaneDetection 
import matplotlib.pyplot as plt
#class object for lane detection
LD = LaneDetection.LaneDetection()
clip = VideoFileClip("project_video.mp4")#.subclip(10,13)
##test 0
out_clip = clip.fx(LaneDetection.LD_pipeline_VidProc, LD,vid=1,visualize=False)
out_clip.write_videofile('project_video_output.mp4', audio=False)    

##test1
'''
count = 0  
for image in clip.iter_frames():
    colored_lane = LaneDetection.LaneDetection_pipline(image,LD,vid=1,visualize=False)
    out_fname = '../output_images/frame_%05d' %count 
    print('left_curv = %s right_curv = %s' %(LD.left_curvature,LD.right_curvature))
    print('detected left pts = %s' %LD.n_detected_pts_left)
    print('detected right pts = %s' %LD.n_detected_pts_right)
    print('detected prev left pts = %s' %LD.n_prev_detected_pts_left)
    print('detected prev right pts = %s' %LD.n_prev_detected_pts_right)
    plt.imshow(colored_lane)
    plt.show()
    #plt.imsave(out_fname,colored_lane)
    count +=1
'''

'''
##test2
    fname = '../test_images/straight_lines1.jpg'
    image = plt.imread(fname)
    colored_lane = LD.LaneDetection_pipline(image)
    plt.imshow(colored_lane)
    plt.show()
'''