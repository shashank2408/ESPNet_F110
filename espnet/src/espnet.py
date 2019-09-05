#!/usr/bin/env python
import rospy 
import torch
import os
import rosparam
import random
from sensor_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from utils import *
import cv2
from argparse import Namespace


class RunSegmentation:
    
    def __init__(self):
        
        rospy.init_node("espnet")
        imageTopic = rosparam.get_param("image_topic")
        publishTopic = rosparam.get_param("pub_topic")
        imageRes= rosparam.get_param("resolution")
        self.resolution = tuple(imageRes.split("x"))
        self.resolution = [int(self.resolution[0]), int(self.resolution[1])]
        modelType = rosparam.get_param("model")
        scale = rosparam.get_param("scale")
        self.bridge = CvBridge()
        args = Namespace(im_size = self.resolution, model = modelType, s = scale)
        self.model = setupSegNet(args)
        rospy.loginfo("Model setup successfully")
        num_gpus = torch.cuda.device_count()
        self.device = 'cuda' if num_gpus >= 1 else 'cpu'
        self.listener(imageTopic,publishTopic)

    def ImageCallback(self,data):
        try:
            img = self.bridge.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            print(e)
        img_out = self.evaluate(img)
        msg = self.bridge.cv2_to_imgmsg(img_out)

        self.pub.publish(msg)


    def evaluate(self,img):
        w, h, ch = img.shape

        img = data_transform(img,tuple(self.resolution))

        img = img.unsqueeze(0)  # add a batch dimension
        img = img.to(self.device)
        img_out = self.model(img)
        img_out = img_out.squeeze(0)  # add a batch dimension
        img_out = img_out.max(0)[1].byte()  # get the label map
        img_out = img_out.to(device='cpu').numpy()
        img_out = relabel(img_out)      
        im_size = tuple((h,w))
        img_out = cv2.resize(img_out, im_size, interpolation=cv2.INTER_CUBIC)	
        return img_out


    def listener(self, imageTopic,publishTopic):
        sub = rospy.Subscriber(imageTopic,Image,self.ImageCallback)
        self.pub = rospy.Publisher(publishTopic,Image, queue_size = 1)
        rospy.loginfo("Topics Subscribed. Segmentation Started")

if __name__ == "__main__":
    obj = RunSegmentation()
    rospy.spin()
