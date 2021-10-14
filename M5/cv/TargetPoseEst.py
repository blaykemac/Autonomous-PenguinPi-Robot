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

# estimate the pose of a target based on size and location of its bounding box in the robot's camera view and the robot's pose
def estimate_pose(base_dir, camera_matrix, completed_img_dict):
    camera_matrix = camera_matrix
    focal_length = camera_matrix[0][0]
    # actual sizes of targets
    target_dimensions = []
    apple_dimensions = [0.075448, 0.074871, 0.071889]
    target_dimensions.append(apple_dimensions)
    lemon_dimensions = [0.060588, 0.059299, 0.053017]
    target_dimensions.append(lemon_dimensions)
    person_dimensions = [0.07112, 0.18796, 0.37592]
    target_dimensions.append(person_dimensions)
    
    target_list = ['apple', 'lemon', 'person']

    target_pose_dict = {}
    
    # for each target in each detection output, estimate its pose
    for target_num in completed_img_dict.keys():
        box = completed_img_dict[target_num]['target'] # [[x],[y],[width],[height]]
        robot_pose = completed_img_dict[target_num]['robot'] # [[x], [y], [theta]]
        true_height = target_dimensions[target_num-1][2]
        
        ######### Replace with your codes #########
        # TODO: compute pose of the target based on bounding box info and robot's pose
        target_pose = {'y': 0.0, 'x': 0.0}
        image_width = 640
        image_height = 480
        
        u0 = image_width / 2
        v0 = image_height / 2
        
        u = box[0][0] - u0
        v = v0 - (box[1][0] + box[3][0] / 2)
         
        uprime = box[0][0] - u0
        vprime = v0 - (box[1][0] - box[3][0] / 2)
        
        OB = np.array([focal_length, u, v])
        
        k = true_height / (vprime - v)
        
        OBprime = k * OB
        OBprime_corrected = OBprime
        OBprime_corrected[1] *= -1
        
        th = robot_pose[2][0]
        robot_xy = np.block([robot_pose[0], robot_pose[1]])
        R_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        lm_bff = OBprime[0:2].T
        OB_world = robot_xy + R_theta @ lm_bff
        
        target_pose['x'] = OB_world[0]
        target_pose['y'] = OB_world[1]
        
        target_pose_dict[target_list[target_num-1]] = target_pose
        
        print(target_pose)
        ###########################################

    return target_pose_dict

# merge the estimations of the targets so that there are at most 3 estimations of each target type
def merge_estimations(target_pose_dict):
    target_pose_dict = target_pose_dict
    apple_est, lemon_est, person_est = [], [], []
    target_est = {}
    
    # combine the estimations from multiple detector outputs
    for f in target_map:
        for key in target_map[f]:
            if key.startswith('apple'):
                apple_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('lemon'):
                lemon_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('person'):
                person_est.append(np.array(list(target_map[f][key].values()), dtype=float))

    ######### Replace with your codes #########
    # TODO: the operation below takes the first three estimations of each target type, replace it with a better merge solution
    
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
    

    for i in range(3):
        try:
            target_est['apple_'+str(i)] = {'y':apple_est[i][0], 'x':apple_est[i][1]}
        except:
            pass
        try:
            target_est['lemon_'+str(i)] = {'y':lemon_est[i][0], 'x':lemon_est[i][1]}
        except:
            pass
        try:
            target_est['person_'+str(i)] = {'y':person_est[i][0], 'x':person_est[i][1]}
        except:
            pass
    ###########################################
        
    return target_est


if __name__ == "__main__":
    # camera_matrix = np.ones((3,3))/2
    fileK = "{}intrinsic.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    base_dir = Path('./')
    
    
    # a dictionary of all the saved detector outputs
    image_poses = {}
    with open(base_dir/'lab_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose']
    
    # estimate pose of targets in each detector output
    target_map = {}        
    for file_path in image_poses.keys():
        completed_img_dict = get_image_info(base_dir, file_path, image_poses)
        target_map[file_path] = estimate_pose(base_dir, camera_matrix, completed_img_dict)

    # merge the estimations of the targets so that there are at most 3 estimations of each target type
    target_est = merge_estimations(target_map)
                     
    # save target pose estimations
    with open(base_dir/'lab_output/targets.txt', 'w') as fo:
        json.dump(target_est, fo)
    
    print('Estimations saved!')



