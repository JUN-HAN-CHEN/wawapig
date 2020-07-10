#!/usr/bin/python
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseWithCovariance, Pose, PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as Image2
import time
import numpy as np
from std_srvs.srv import Trigger, TriggerResponse
img = None
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os
import math
i=0
fp = None
import argparse
path = "/media/jun-han-chen/1cdbfaff-5bf1-4b8a-b396-cde848807c08/Bayes/datasets/"
parser = argparse.ArgumentParser()
parser.add_argument('--dir', required=True)
args = parser.parse_args()
print(args.dir)
fpath = os.path.join(path, args.dir)
flag= False
x, y, z, qx, qy, qz, qw = 0, 0, 0, 0, 0, 0, 0

def take():
    global img,i
    bridge = CvBridge()
    image_np = bridge.imgmsg_to_cv2(img, "rgb8")
    image_np=Image.fromarray(image_np,"RGB")
    plt.imshow(image_np)
    plt.axis('off')
    plt.savefig(fpath+ '/' +'train_'+str(i)+".png")
    print(i)	
def rgb_callback(image):
    global img
    img=image
#    print(img)


def callback(msg):
    global fp, i,img
    amclpose = PoseStamped()
    amclpose = msg
    #print("fuck")
    # print("--------------------------")
    # print(img.header.stamp.secs)
    # print(img.header.stamp.nsecs)
    # print(amclpose)
    # print(amclpose.header.stamp.secs)
    # print(amclpose.header.stamp.nsecs)
    global x, y, z, qx, qy, qz, qw
    global flag
    flag=True
    # if( math.sqrt( (x-amclpose.pose.position.x)**2 + (y-amclpose.position.y)**2) > 0.01 or math.sqrt( (amclpose.orientation.w - qw)**2 + (amclpose.orientation.z - qz)**2) > 0.01):
    
    #     flag = True
        # print(math.sqrt( (x-amclpose.position.x)**2 + (y-amclpose.position.y)**2), math.sqrt( (x-amclpose.position.x)**2 + (y-amclpose.position.y)**2) > 0.01, math.sqrt( (amclpose.orientation.w - qw)**2 + (amclpose.orientation.z - qz)**2) > 0.0001)
        # print(flag)
        # print(x, y, qw, qz)
    if(img != None):
        if flag:
            fp.write("train_"+str(i)+".png")
            take()
            i=i+1	
            fp.write(" ")
            fp.write(str(amclpose.pose.position.x))
            fp.write(" ")
            fp.write(str(amclpose.pose.position.y))
            fp.write(" ")
            fp.write(str(amclpose.pose.position.z))
            fp.write(" ")
            fp.write(str(amclpose.pose.orientation.x))
            fp.write(" ")
            fp.write(str(amclpose.pose.orientation.y))
            fp.write(" ")
            fp.write(str(amclpose.pose.orientation.z))
            fp.write(" ")
            fp.write(str(amclpose.pose.orientation.w))
            fp.write("\n")
            # x, y, qz, qw = amclpose.position.x, amclpose.position.y, amclpose.orientation.z, amclpose.orientation.w
            flag = False
            # print(flag)
            # print (amclpose.position.x, amclpose.position.y, amclpose.orientation.z, amclpose.orientation.w)
try:
    os.mkdir(fpath)
except:
    pass
filename = os.path.join(fpath, 'dataset_train.txt')
fp = open(filename, "w")
fp.write(time.asctime(time.localtime(time.time())))
fp.write("\n")
fp.write(args.dir)
fp.write("\n")
fp.write("\n")
rospy.init_node('check_odometry')
rospy.Subscriber("/image_raw",Image2,rgb_callback)
rospy.Subscriber('/current_pose', PoseStamped, callback)
#rospy.Subscriber("/camera/rgb/image_raw",Image2,rgb_callback)
rospy.spin()