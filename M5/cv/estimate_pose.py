# estimate the pose of a target object detected
import numpy as np
import json
import os
from pathlib import Path
import ast
import cv2
import math
from machinevisiontoolbox import Image

import matplotlib.pyplot as plt
import PIL

# use the machinevision toolbox to get the bounding box of the detected target(s) in an image
def get_bounding_box(target_number, image_path):
    image = PIL.Image.open(image_path).resize((640,480), PIL.Image.NEAREST)
    target = Image(image)==target_number
    blobs = target.blobs()
    [[u1,u2],[v1,v2]] = blobs[0].bbox # bounding box
    width = abs(u1-u2)
    height = abs(v1-v2)
    center = np.array(blobs[0].centroid).reshape(2,)
    box = [center[0], center[1], int(width), int(height)] # box=[x,y,width,height]
    # plt.imshow(fruit.image)
    # plt.annotate(str(fruit_number), np.array(blobs[0].centroid).reshape(2,))
    # plt.show()
    # assert len(blobs) == 1, "An image should contain only one object of each target type"
    return box

# read in the list of detection results with bounding boxes and their matching robot pose info
def get_image_info(base_dir, file_path, image_poses):
    # there are at most three types of targets in each image
    target_lst_box = [[], [], []]
    target_lst_pose = [[], [], []]
    completed_img_dict = {}

    # add the bounding box info of each target in each image
    # target labels: 1 = apple, 2 = lemon, 3 = person, 0 = not_a_target
    img_vals = set(Image(base_dir / file_path, grey=True).image.reshape(-1))
    for target_num in img_vals:
        if target_num > 0:
            try:
                box = get_bounding_box(target_num, base_dir/file_path) # [x,y,width,height]
                pose = image_poses[file_path] # [x, y, theta]
                target_lst_box[target_num-1].append(box) # bouncing box of target
                target_lst_pose[target_num-1].append(np.array(pose).reshape(3,)) # robot pose
            except ZeroDivisionError:
                pass

    # if there are more than one objects of the same type, combine them
    for i in range(3):
        if len(target_lst_box[i])>0:
            box = np.stack(target_lst_box[i], axis=1)
            pose = np.stack(target_lst_pose[i], axis=1)
            completed_img_dict[i+1] = {'target': box, 'robot': pose}
        
    return completed_img_dict

def box_to_world(box, robot_pose, class_name, focal_length):
        
    ######### Replace with your codes #########
    # TODO: compute pose of the target based on bounding box info and robot's pose
    # actual sizes of targets
    apple_dimensions = [0.075448, 0.074871, 0.071889]
    lemon_dimensions = [0.060588, 0.059299, 0.053017]
    person_dimensions = [0.07112, 0.18796, 0.37592]
    target_dimensions = {"apple": apple_dimensions, "lemon": lemon_dimensions, "person": person_dimensions}
    true_height = target_dimensions[class_name][2]
    
    target_pose = {'y': 0.0, 'x': 0.0}
    image_width = 640
    image_height = 480
    
    # image centre coordiante system origin
    u0 = image_width / 2
    v0 = image_height / 2
    
    # left of the box
    u = box[0] - u0
    # bottom of box
    v = v0 - (box[1] + box[3] / 2)
     # right of box
    uprime = box[0] - u0
    # top of box
    vprime = v0 - (box[1] - box[3] / 2)
    
    OB = np.array([focal_length, u, v])
    
    k = true_height / (vprime - v)
    
    OBprime = k * OB
    OBprime_corrected = OBprime
    OBprime_corrected[1] *= -1
    
    th = robot_pose[2][0]
    robot_xy = np.block([robot_pose[0], robot_pose[1]])
    R_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    lm_bff = OBprime[0:2].T
    camera_offset = np.array([0.08, 0])
    OB_world = robot_xy + camera_offset + R_theta @ lm_bff
    
    return np.array([OB_world[0], OB_world[1]]) #(x,y), world coordinates


# merge the estimations of the targets so that there are at most 3 estimations of each target type
def merge_estimations(objects_est):
    apple_est, lemon_est, person_est = objects_est[0], objects_est[1], objects_est[2]

    ######### Replace with your codes #########
    # TODO: the operation below takes the first three estimations of each target type, replace it with a better merge solution
    
    """
    # do not run kmeans if we only have images of one distinct object of a given class
    for i, apple in enumerate(apple_est):
        for j, other_apple in enumerate(apple_est):
            if i == j:
                continue
            else:
                radius = np.hypot(apple[0] - other_apple[0] , apple[1] - other_apple[1])
                if radius < merge_threshold:
                    dup_indices.append(j)
    """
    
    # this only works assuming that if there are less than or equal to 3 images, they are all distinct objects
    k_apple = min(len(apple_est), 3)
    k_lemon = min(len(lemon_est), 3)
    k_person = min(len(person_est), 3)
    
    epsilon = 1E-3
    k = 3
    attempts = 10
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, 20, epsilon)
    
    flags = cv2.KMEANS_RANDOM_CENTERS

    if len(apple_est) > 1:
        apple_est = np.array(apple_est)
        apple_est = np.float32(apple_est)
        compactness, labels, centers = cv2.kmeans(apple_est, k_apple, None, criteria, attempts, flags)
        apple_est = np.float64(centers)

    if len(lemon_est) > 1:
        lemon_est = np.array(lemon_est)
        lemon_est = np.float32(lemon_est)
        compactness, labels, centers = cv2.kmeans(lemon_est, k_lemon, None, criteria, attempts, flags)
        lemon_est = np.float64(centers)
    
    if len(person_est) > 1:
        person_est = np.array(person_est)
        person_est = np.float32(person_est)
        compactness, labels, centers = cv2.kmeans(person_est, k_person, None, criteria, attempts, flags)
        person_est = np.float64(centers)
    
    target_est_dict = {}

    for i in range(3):
        try:
            target_est_dict['apple_'+str(i)] = {'y':apple_est[i][1], 'x':apple_est[i][0]}
        except:
            pass
        try:
            target_est_dict['lemon_'+str(i)] = {'y':lemon_est[i][1], 'x':lemon_est[i][0]}
        except:
            pass
        try:
            target_est_dict['person_'+str(i)] = {'y':person_est[i][1], 'x':person_est[i][0]}
        except:
            pass
            
    target_est_list = [apple_est, lemon_est, person_est]
    ###########################################
        
    return target_est_dict, target_est_list
