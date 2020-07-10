#! /usr/bin/env python

import rospy
from sensor_msgs.msg import Image as Image2
from sensor_msgs.msg import LaserScan
import message_filters
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseWithCovariance, Pose, PoseStamped
from PIL import Image
import matplotlib.pyplot as plt
from cv_bridge import CvBridge, CvBridgeError
import time
import math
import os
import sys
path = "/media/jun-han-chen/1cdbfaff-5bf1-4b8a-b396-cde848807c08/Bayes/datasets/village/"
fpath = '/media/jun-han-chen/1cdbfaff-5bf1-4b8a-b396-cde848807c08/Bayes/datasets/village/sync'
i=0
class sync():
    
    def __init__(self):
        
        image_sub = message_filters.Subscriber("/image_raw", Image2)
        pose_sub = message_filters.Subscriber("/current_pose", PoseStamped)
        
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, pose_sub], queue_size = 10, slop = 0.1)
        ts.registerCallback(self.callback)
        
        
    def callback(self, image_raw, current_pose):
        global i
        bridge = CvBridge()
        image_np = bridge.imgmsg_to_cv2(image_raw, "rgb8")
        image_np=Image.fromarray(image_np,"RGB")
        plt.imshow(image_np)
        plt.axis('off')
        plt.savefig(fpath+ '/' +'train_'+str(i)+".png")
        print(i)
        i+=1
        # rospy.loginfo(image_raw.header.stamp)
        # rospy.loginfo(current_pose.header.stamp)
        fp.write("train_"+str(i)+".png")
        fp.write(" ")
        fp.write(str(current_pose.pose.position.x))
        fp.write(" ")
        fp.write(str(current_pose.pose.position.y))
        fp.write(" ")
        fp.write(str(current_pose.pose.position.z))
        fp.write(" ")
        fp.write(str(current_pose.pose.orientation.x))
        fp.write(" ")
        fp.write(str(current_pose.pose.orientation.y))
        fp.write(" ")
        fp.write(str(current_pose.pose.orientation.z))
        fp.write(" ")
        fp.write(str(current_pose.pose.orientation.w))
        fp.write("\n")

if __name__ == "__main__":
    filename = os.path.join(path, 'dataset_train.txt')
    fp = open(filename, "w")
    fp.write("Visual Localization Dataset village")
    fp.write("\n")
    fp.write('ImageFile, Camera Position [X Y Z W P Q R]')
    fp.write("\n")
    fp.write("\n")
    rospy.init_node("sync")
    
    sync()
    
    rospy.spin()