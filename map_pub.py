#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pcl
import rospy
import rospy
from sensor_msgs.msg import PointField
from sensor_msgs.msg import PointCloud2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import Image
# import cv2
from numpy import save
# from tf.transformations import *

class Ceiling_map_pub:
    def __init__(self,map_root):
        rospy.init_node("ceiling_map_pub")
        self.pub = rospy.Publisher('ceiling/map', PointCloud2, queue_size=1)
        self.map_root = map_root
        self.ceiling_map = pcl.load(self.map_root)
        self.resolution = 0.2
        voxel = self.ceiling_map.make_voxel_grid_filter()
        voxel.set_leaf_size(self.resolution,self.resolution,self.resolution)
        self.map_filtered = voxel.filter()
        self.map_filtered = np.asarray(self.map_filtered)
        

    def xyz_array_to_pointcloud2(self, points, stamp=None, frame_id=None):
        msg = PointCloud2()
        if stamp:
            msg.header.stamp = stamp
        if frame_id:
            msg.header.frame_id = frame_id
        if len(points.shape) == 3:
            msg.height = points.shape[1]
            msg.width = points.shape[0]
        else:
            msg.height = 1
            msg.width = len(points)
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12*points.shape[0]
        msg.is_dense = int(np.isfinite(points).all())
        msg.data = np.asarray(points, np.float32).tostring()
        return msg

    def map_publish(self, data):
        # ceiling_map = pcl.load(self.map_root)
        # ceiling_map = np.asarray(ceiling_map)
        msg = self.xyz_array_to_pointcloud2(data, frame_id='/map')
        self.pub.publish(msg)

if __name__ == '__main__':
    print('map_publish')
    map_root = "/media/jun-han-chen/1cdbfaff-5bf1-4b8a-b396-cde848807c08/Bayes/datasets/village/point_cloud_map.pcd"
    ceiling_map = Ceiling_map_pub(map_root)
    print("size of map:", ceiling_map.map_filtered.shape)
    while not rospy.is_shutdown():   
        ceiling_map.map_publish(ceiling_map.map_filtered)