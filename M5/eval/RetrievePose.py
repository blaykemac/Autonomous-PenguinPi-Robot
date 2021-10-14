#!/usr/bin/env python

# get the pose of objects in Gazebo after the fruit delivery is done
import sys
import math
import os
import random
import json
from math import pi

import copy
import rospy
import tf
from gazebo_msgs.srv import GetModelState
from geometry_msgs.msg import *
from geometry_msgs.msg import Point, Quaternion
import numpy as np

import rospkg

rospack = rospkg.RosPack()

class SceneManager:
    def __init__(self):
        self.obj_class_dict = {
				'aruco1':1, 'aruco2':1, 'aruco3':1, 'aruco4':1, 'aruco5':1, 
				'aruco6':1, 'aruco7':1, 'aruco8':1, 'aruco9':1, 'aruco10':1,
				'person':3, 'apple':3, 'lemon':3}
        print(self.obj_class_dict)
        self.workspace_path =  rospack.get_path('penguinpi_gazebo')

    def save_all_objs(self):
        print('Waiting for gazebo services...')
        model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        obj_counter = 0
        map_dict = {}
        for key in self.obj_class_dict.keys():
            for i in range(0, int(self.obj_class_dict[key])):
                obj_counter += 1
                item_name = '%s_%i' % (key, i)
                resp_coordinates = model_coordinates(item_name,'')
                map_dict[item_name] = {'x': resp_coordinates.pose.position.x, 'y': resp_coordinates.pose.position.y}

        f_path = self.workspace_path + '/layout_after_delivery.txt'
        with open(f_path, 'w') as f:
            json.dump(map_dict, f)


if __name__ == '__main__':
    obj_manager = SceneManager()
    obj_manager.save_all_objs()

