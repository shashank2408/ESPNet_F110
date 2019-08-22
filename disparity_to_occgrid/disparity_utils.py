import numpy as np
import cv2
from image_geometry.cameramodels import StereoCameraModel
from cv_bridge import CvBridge, CvBridgeError

stereoObj = StereoCameraModel()
bridge = CvBridge()

def getIndicesOfObjects(img, objects):
    """ Iterates through the objects list and gets list of all indices 
        where the pixel value equals the object label"""
    
    
    print("Getting indices")
    indices =None
    x=[]
    y=[]
    
    for objNum in objects:
        indices = np.array(np.where(img == objNum))
        x = np.hstack((x,indices[0]))
       	y = np.hstack((y,indices[1]))
    x= x[:,np.newaxis]
    y= y[:,np.newaxis]
    cords = np.hstack((x,y))
    return cords

def isValid(point):
	a = np.isfinite(point)
	if False in a:
		return False
	return True

def getPointsTo3D(cords, disparityImage,leftCamMsg,rightCamMsg):
    print("getting points to 3d")
    stereoObj.fromCameraInfo(leftCamMsg,rightCamMsg)
    
    disparityIm = bridge.imgmsg_to_cv2(disparityImage.image)
 
    disparity = np.interp(disparityIm,(disparityImage.min_disparity,disparityImage.max_disparity),(0,255))
    points = list(map(lambda point: get3DCoord(point,disparity[int(point[0]),int(point[1])]),cords))
    points = list(filter(lambda pt: isValid(pt), points))
    points = list(map(lambda pt: imgToGrid(pt), points))

    return points

def get3DCoord(coord, disparity):


    #index = np.ravel_multi_index((int(coord[0]),int(coord[1])),(imHeight,imWidth))
    #if np.isfinite(disparity[int(coord[0]), int(coord[1])]):
	#disp_val=disparity[int(coord[0]), int(coord[1])]-min_disp
    #else:
    #	disp_val = 0
    #print(coord,disparity) 
    point =  stereoObj.projectPixelTo3d((coord[0],coord[1]),np.float(disparity))  
    return point


def imgToGrid(point):
    #print(point)
    x=point[0]*200
    y=point[1]*200

    x=round(x)
    y=round(y)

    #y = y+500;
    y = 1000 if y>1000 else y
    y = 0 if y <0 else y
    x=x+500
    x = 1000 if x>1000 else x
    x = 0 if x < 0 else x 
    return tuple((x,y))




# file = "/home/mlab-train/Desktop/deeprl/LaneDetection/EdgeNets/results_city_test/results/image_right000001.png"

# img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)

# objects = [11,21]
# cords = getIndicesOfObjects(img,objects)
# print (cords)
