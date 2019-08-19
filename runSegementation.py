import rospy 
import torch
import os
import rosparam
from sensor_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from utils import *
import cv2
from argparse import Namespace
class RunSegmentation:
    
    def __init__(self):
        imageTopic = rosparam.get_param("image_topic")
        publishTopic = rosparam.get_param("pub_topic")
        imageRes= rosparam.get_param("resolution")
        self.resolution = imageRes.split("x")
        self.resolution = [ int(self.resolution[0]), int(self.resolution[1])]
        modelType = rosparam.get_param("model")
        scale = rosparam.get_param("scale")
        self.bridge = CvBridge()
        args = Namespace(im_size = self.resolution, model = modelType, s = scale)

        self.model = setupSegNet(args)
        self.listener(imageTopic,publishTopic)


    def ImageCallback(self,data):
        try:
            img = self.bridge.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            print(e)
        img_out = self.evaluate(img)
        self.pub.publish(img_out)


    def evaluate(self,img):
        w, h = cv2.GetSize(img)
        device = 'cuda' if num_gpus >= 1 else 'cpu'
        img = data_transfom(img,tuple(self.resolution))
        img = img.unsqueeze(0)  # add a batch dimension
        img = img.to(device)
        img_out = model(img)
        img_out = img_out.max(0)[1].byte()  # get the label map
        img_out = img_out.to(device='cpu').numpy()
        img_out = relabel(img_out)
        
        img_out = cv2.resize(img_out, (w, h), interpolation=cv2.INTER_NEAREST)

        return img_out


    def listener(self, imageTopic,publishTopic):
        rospy.init_node("ESPNet_ROS")
        sub = rospy.Subscriber(imageTopic,Image,ImageCallback)
        self.pub = rospy.Publisher(publishTopic,Image, queue_size = 10)


if __name__ == "__main__":
    obj = RunSegmentation()
    rospy.spin()
