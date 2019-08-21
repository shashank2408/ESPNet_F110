import numpy as np
import cv2


def getIndicesOfObjects(img, objects):
    """ Iterates through the objects list and gets list of all indices 
        where the pixel value equals the object label"""

    x = []
    y = []

    for objNum in objects:
        indices = np.array(np.where(img == objNum))
        x = np.hstack((x,indices[0]))
        y = np.hstack((y,indices[1]))
    x= x[:,np.newaxis]
    y= y[:,np.newaxis]
    cords = np.hstack((x,y))
    return cords

def getPointsTo3D(cords, disparityImage):
    stereoObj = StereoCameraModel()
    stereoObj.fromCameraInfo(self.leftCamInfo,self.rightCamInfo)
    points = list(map(lambda point: get3DCoord(stereoObj,point,disparityImage[point[0],point[1]],cords)))

def get3DCoord(stereoObj, coord, disparity):
    return stereoObj.projectPixelTo3d((coord[0],coord[1]),disparity)


def imgToGrid(point):
    x=point[0]*200
    y=point[1]*200
    x=round(x)
    y=round(y)
    y = y+500;
    y = y>1000?1000:y;
    y = y<0?0:y;
    x = x>1000?1000:x;
    return tuple((x,y))




# file = "/home/mlab-train/Desktop/deeprl/LaneDetection/EdgeNets/results_city_test/results/image_right000001.png"

# img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)

# objects = [11,21]
# cords = getIndicesOfObjects(img,objects)
# print (cords)
