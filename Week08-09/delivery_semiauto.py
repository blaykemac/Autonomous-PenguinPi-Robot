# semi-automatic approach for fruit delivery

# import modules
import sys, os
import ast
import numpy as np
import json
import argparse

sys.path.insert(0, "../util")
from util.pibot import PenguinPi

# read in the object poses, note that the object pose array is [y,x]
def parse_map(fname: str) -> dict:
    with open(fname,'r') as f:
        gt_dict = ast.literal_eval(f.readline())        
        apple_gt, lemon_gt, person_gt = [], [], []

        # remove unique id of targets of the same type 
        for key in gt_dict:
            if key.startswith('apple'):
                apple_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
            elif key.startswith('lemon'):
                lemon_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
            elif key.startswith('person'):
                person_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
    
    # if more than 3 estimations are given for a target type, only the first 3 estimations will be used
    if len(apple_gt) > 3:
        apple_gt = apple_gt[0:3]
    if len(lemon_gt) > 3:
        lemon_gt = lemon_gt[0:3]
    if len(person_gt) > 3:
        person_egt = person_gt[0:3]
    
    return apple_gt, lemon_gt, person_gt

# find lemons too close to person and apples too far from person using Euclidean distance (threshold = 0.5)
def compute_dist(apple_list, lemon_list, person_list):
    apple_list = apple_list
    lemon_list = lemon_list
    person_list = person_list
    to_move = {}
    i = 0
    for person in person_list:
        to_move['person_'+str(i)] = {}
        j,k = 0,0
        # find apples that are too far
        for apple in apple_list:
            if abs(np.linalg.norm(apple-person)) > 0.5:
                to_move['person_'+str(i)]['apple_'+str(j)] = apple
                to_move['person_'+str(i)]['dist_'+str(j)] = abs(np.linalg.norm(apple-person))
                j = j+1
            try: 
                to_move['person_'+str(i)]['apple_2']
                print('All apples too far from Person:', person)
            except:
                pass
        # find lemons that are too close
        for lemon in lemon_list:
            if abs(np.linalg.norm(lemon-person)) < 0.5:
                to_move['person_'+str(i)]['lemon_'+str(k)] = lemon
                to_move['person_'+str(i)]['dist_'+str(k)] = abs(np.linalg.norm(lemon-person))
                print('There are lemons too close to Person:', person)
            k = k+1
        i = i+1
    return to_move

# semi-automatic delivery approach by providing a series of waypoints to guide the robot to move all targets
# the robot automatically drives to a given [x,y] coordinate
# additional improvements:
# you may use different motion model parameters for robot driving on its own or driving while pushing a fruit
# try changing to a fully automatic delivery approach: develop a path-finding algorithm that produces the waypoints
def drive_to_point(waypoint, robot_pose):
    waypoint = waypoint
    robot_pose = robot_pose
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    wheel_vel = 30 # tick ?? per second ??
    turning_vel = 5
    
    # turn towards the waypoint
    angle_to_waypoint = np.arctan2((waypoint[1] - robot_pose[1]) , (waypoint[0] - robot_pose[0]))
    delta_angle = np.arctan2(np.sin(angle_to_waypoint - robot_pose[2]), np.cos(angle_to_waypoint - robot_pose[2]))
    modified_wheel_vel = turning_vel
    if delta_angle < 0:
        modified_wheel_vel *= -1
    distance_turning = baseline / 2 * abs(delta_angle)
    turn_time = distance_turning / (scale * turning_vel)
    print("Turning for {:.2f} seconds".format(turn_time))
    ppi.set_velocity([0, 1], turning_tick=modified_wheel_vel, time=turn_time)
    
    # after turning, drive straight to the waypoint
    distance_straight = np.sqrt((waypoint[0] - robot_pose[0])**2 + (waypoint[1] - robot_pose[1])**2)
    drive_time = distance_straight / (scale * wheel_vel) # replace with your calculation
    print("Driving for {:.2f} seconds".format(drive_time))
    ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)

    # update the robot pose [x,y,theta]
    robot_pose = [waypoint[0], waypoint[1], angle_to_waypoint] 
    ####################################################
    return robot_pose

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit delivery")
    parser.add_argument("--map", type=str, default='M4_marking_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    args, _ = parser.parse_known_args()

    ppi = PenguinPi(args.ip,args.port)

    # read in the map
    apple_gt, lemon_gt, person_gt = parse_map(args.map)
    print("Map: apple = {}, lemon = {}, person = {}".format(apple_gt, lemon_gt, person_gt))

    # find apple(s) and lemon(s) that need to be moved
    to_move = compute_dist(apple_gt, lemon_gt, person_gt)
    print("Fruits to be moved: ", to_move)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]
    # semi-automatic approach for fruit delivery
    while True:
        # enter the waypoints
        # instead of manually enter waypoints, you can get coordinates by clicking on a map, see camera_calibration.py
        x,y = 0.0,0.0
        x = input("X coordinate of the waypoint: ")
        try:
            x = float(x)
        except ValueError:
            print("Please enter a number.")
            continue
        y = input("Y coordinate of the waypoint: ")
        try:
            y = float(y)
        except ValueError:
            print("Please enter a number.")
            continue
        
        # robot drives to the waypoint
        waypoint = [x,y]
        robot_pose = drive_to_point(waypoint,robot_pose)
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

        # exit
        ppi.set_velocity([0, 0])
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput == 'N':
            break