from disparity_utils import *
from stereo_msgs.msg import DisparityImage
import rospy 
import numba
import rosparam
from image_geometry.cameramodels import StereoCameraModel
from sensor_msgs.msg import *



segSub = rosparam.get_param("segmented_topic")
dispSub = rosparam.get_param("disparity_topic")
layersFile = rosparam.get_param("layers_File")


leftCamInfo = rosparam.get_param("left_cam_info")
rightCamInfo = rosparam.get_param("right_cam_info")


leftCamMsg = rospy.wait_for_message(leftCamInfo, CameraInfo)
rightCamMsg = rospy.wait_for_message(rightCamInfo, CameraInfo)


stereoObj = StereoCameraModel()
print("Cloading left and right")
stereoObj.fromCameraInfo(leftCamMsg,rightCamMsg)

coord = (141.0, 326.0)
disparity= -35.58024
print("Computing point")
point =  stereoObj.projectPixelTo3d((coord[0],coord[1]),np.float(disparity))

print(point)
