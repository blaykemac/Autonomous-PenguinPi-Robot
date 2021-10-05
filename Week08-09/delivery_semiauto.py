# semi-automatic approach for fruit delivery

# import modules
import sys, os
import ast
import numpy as np
import json
import argparse

import pygame

import cv2 
import time

# import utility functions
sys.path.insert(0, "{}/util".format(os.getcwd()))
from util.pibot import PenguinPi # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
import shutil # python package for file operations

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

import math

RED = (255, 0, 0) # apple
YELLOW = (255, 255, 0) #lemon
WHITE = (255,  255, 255) #background
BLUE = (0, 0, 255) #aruco
BLACK = (0, 0, 0) #person


class OperateModified:
    def __init__(self, args, markers):
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)
        
        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip, markers)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length = 0.07) # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.bg = pygame.image.load('pics/gui_mask.jpg')

    # wheel control
    def control(self):       
        if args.play_data:
            lv, rv = self.pibot.set_velocity()            
        else:
            print(f"cmd: {self.command['motion']}")
            lv, rv = self.pibot.set_velocity(
                self.command['motion'])
            print(f"lv, rv: {lv}, {rv}")
        if not self.data is None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        drive_meas = measure.Drive(lv, rv, dt)
        #drive_meas = measure.Drive(lv, rv, dt)
        self.control_clock = time.time()
        return drive_meas
    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()
        if not self.data is None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            #self.ekf.add_landmarks(lms) # Don't think we need this to add landmarks if we know them already
            self.ekf.update(lms)


    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip, markers):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)  
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot, markers)

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False
            
        # keyboard teleoperation        
    def update_keyboard(self):
        for event in pygame.event.get():
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'][0] = min(self.command['motion'][0]+1, 1)
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'][0] = max(self.command['motion'][0]-1, -1)
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'][1] = min(self.command['motion'][1]+1, 1)
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'][1] = max(self.command['motion'][1]-1, -1)
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm +=1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'
            # run object detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['inference'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_inference'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()


# read in the object poses, note that the object pose array is [y,x]
def parse_map(fname: str) -> dict:
    with open(fname,'r') as f:
        gt_dict = ast.literal_eval(f.readline())        
        apple_gt, lemon_gt, person_gt, aruco_gt = [], [], [], []
        
        # fill aruco_gt list with 10 blank coordinates
        aruco_gt = [np.array([0,0])] * 10
        #print(f"aruco_gt: {aruco_gt}")
        
        #delete later
        #print(f"gt_dict: {gt_dict}")

        # remove unique id of targets of the same type 
        for key in gt_dict:
        
            if key.startswith('apple'):
                apple_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
            elif key.startswith('lemon'):
                lemon_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
            elif key.startswith('person'):
                person_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
                
            elif key.startswith('aruco'): #aruco10_0, aruco7_0
                marker_index = int(key[5:].split('_')[0]) - 1 # subtract 1 to account for offset in aruco names (aruco1 to aruco10)
                aruco_gt[marker_index] = (np.array(list(gt_dict[key].values()), dtype=float))
    
    # if more than 3 estimations are given for a target type, only the first 3 estimations will be used
    if len(apple_gt) > 3:
        apple_gt = apple_gt[0:3]
    if len(lemon_gt) > 3:
        lemon_gt = lemon_gt[0:3]
    if len(person_gt) > 3:
        person_gt = person_gt[0:3]
    
    #print(f"aruco_gt: {aruco_gt}")
    
    return apple_gt, lemon_gt, person_gt, aruco_gt

# find lemons too close to person and apples too far from person using Euclidean distance (threshold = 0.5)
def compute_dist(apple_list, lemon_list, person_list):
    apple_list = apple_list
    lemon_list = lemon_list
    person_list = person_list
    
    #delete later
    print(f"person_list: {person_list}")
    
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
    print(f"to_move: {to_move}")
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
    
    #########################################
    
    
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

def slam_to_point(waypoint, operate):
    robot_pose = operate.ekf.robot.state
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    #########################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    wheel_vel = 30 # tick ?? per second ??
    turning_vel = 5
    
    ##################################################
    waypoint = np.array(waypoint)
    
    goal_theta = np.arctan2((waypoint[1] - robot_pose[1]) , (waypoint[0] - robot_pose[0]))
    K_pt = 1
    tolerance = 0.1 # rads
    control_omega, error_theta = PControllerOmega(goal_theta, operate.ekf.robot.state[2], K_pt)
    
    
    # Enter control loop to align the theta pose
    while abs(error_theta) > tolerance:
        control_omega, error_theta = PControllerOmega(goal_theta, operate.ekf.robot.state[2], K_pt)
        control_v = 0
        
        print(f"omega: {control_omega}")
        print(f"error: {error_theta}")
        print(f"robot_state_ekf: {operate.ekf.robot.state}")
    
        operate.command['motion'] = createMotionCommand(control_v, control_omega)
        #operate.command['motion'] = [0, int(control_omega / float(abs(control_omega)))]
        
        print(f"cmd: {operate.command['motion']}")

        #operate.update_keyboard()
        operate.take_pic()
        drive_meas = operate.control()
        
        operate.update_slam(drive_meas)
        
    # Enter control loop to drive straight to correct x,y coordinate, maintaining the same theta.
    stop_criteria_met = False
    
    K_pw = 2
    K_pv = 1
    threshold = 0.1
    
    #print(f"waypoint: {}")
    #print(f"robot state: {operate.ekf.robot.state}")
    
    error_dist = get_distance_robot_to_goal(waypoint, operate.ekf.robot.state)
    goal_theta = get_angle_robot_to_goal(waypoint, operate.ekf.robot.state)
    while not stop_criteria_met:
        if error_dist > threshold:
            control_v, _ = PControllerV(waypoint, operate.ekf.robot.state, K_pv)
            control_omega, _ = PControllerOmegaDynamic(waypoint, operate.ekf.robot.state, K_pw)
            
            print(f"omega: {control_omega}")
            print(f"v: {control_v}")
            print(f"robot_state_ekf: {operate.ekf.robot.state}")
            
            #operate.command['motion'] = [int(control_v / float(abs(control_v))), int(control_omega / float(abs(control_omega)))]
            operate.command['motion'] = createMotionCommand(control_v, control_omega)
       
            '''
            Only run this if we want to go to a desired theta upon x,y arrival at waypoint 
            else:
            v_k = 0
            w_k = 
            '''
            operate.take_pic()
            drive_meas = operate.control()
            operate.update_slam(drive_meas)
        
        else:
            operate.command['motion'] = [0,0]
            operate.control()
            stop_criteria_met = True
        # Now drive using thse controls
        
        
        # Finally update the errors
        error_dist = get_distance_robot_to_goal(waypoint, operate.ekf.robot.state)
        
    
    ##################################################
def createMotionCommand(control_v, control_omega):
    if control_v == 0 and control_omega == 0:
        return [0,0]
    if control_v == 0 and not control_omega == 0:
        return [0, int(control_omega / abs(control_omega) * math.ceil(abs(control_omega)))]         
    if not control_v == 0 and control_omega == 0:
        return [int(control_v / abs(control_v) * math.ceil(abs(control_v))), 0]


            
    return [int(control_v / abs(control_v) * math.ceil(abs(control_v))), int(control_omega / abs(control_omega) * math.ceil(abs(control_omega)))]
    
    
def PControllerOmega(goal_theta, robot_theta, gain):
    error_theta = np.arctan2(np.sin(goal_theta - robot_theta), np.cos(goal_theta - robot_theta))
    control_signal = gain * error_theta
    return float(control_signal), float(error_theta)
    
def PControllerOmegaDynamic(waypoint, robot_state, gain):
    error_theta = get_angle_robot_to_goal(waypoint, robot_state)
    control_signal = gain * error_theta
    return float(control_signal), float(error_theta)
    
def PControllerV(waypoint, robot_state, gain):
    error_radius = get_distance_robot_to_goal(waypoint, robot_state)
    control_signal = gain * error_radius
    return float(control_signal), float(error_radius)
    
def get_distance_robot_to_goal(goal=np.zeros(3), robot_state=np.zeros(3)):
	"""
	Compute Euclidean distance between the robot and the goal location
	:param robot_state: 3D vector (x, y, theta) representing the current state of the robot
	:param goal: 3D Cartesian coordinates of goal location
	"""

	if goal.shape[0] < 3:
		goal = np.hstack((goal, np.array([0])))

	x_goal, y_goal,_ = goal
	x, y,_ = robot_state
	x_diff = x_goal - x
	y_diff = y_goal - y

	rho = np.hypot(x_diff, y_diff)

	return rho

def get_angle_robot_to_goal(goal=np.zeros(3), robot_state=np.zeros(3)):
	"""
	Compute angle to the goal relative to the heading of the robot.
	Angle is restricted to the [-pi, pi] interval
	:param robot_state: 3D vector (x, y, theta) representing the current state of the robot
	:param goal: 3D Cartesian coordinates of goal location
	"""

	if goal.shape[0] < 3:
		goal = np.hstack((goal, np.array([0])))

	x_goal, y_goal,_ = goal
	x, y, theta = robot_state
	x_diff = x_goal - x
	y_diff = y_goal - y

	alpha = clamp_angle(np.arctan2(y_diff, x_diff) - theta)

	return alpha
    
def clamp_angle(rad_angle=0, min_value=-np.pi, max_value=np.pi):
	"""
	Restrict angle to the range [min, max]
	:param rad_angle: angle in radians
	:param min_value: min angle value
	:param max_value: max angle value
	"""

	if min_value > 0:
		min_value *= -1

	angle = (rad_angle + max_value) % (2 * np.pi) + min_value

	return angle
    
# convert pixel coordinates to actual coordinates
def pix_to_world(u,v):
    x = (u - u0) * 3 / width
    y  = (v0 - v) * 3 / width
    return x,y
    
def world_to_pix(x,y):
    u = x * width / 3 + u0
    v = v0 - y * width / 3 
    return u,v

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit delivery")
    parser.add_argument("--map", type=str, default='M4_marking_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--ckpt", default='network/scripts/model/model.best.pth')
    parser.add_argument("--gui", action='store_true')
    args, _ = parser.parse_known_args()

    #ppi = PenguinPi(args.ip,args.port)

    # read in the map
    apple_gt, lemon_gt, person_gt, aruco_gt = parse_map(args.map)
    print("Map: apple = {}, lemon = {}, person = {}".format(apple_gt, lemon_gt, person_gt))
    print(aruco_gt)


    # find apple(s) and lemon(s) that need to be moved
    to_move = compute_dist(apple_gt, lemon_gt, person_gt)
    print("Fruits to be moved: ", to_move)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]
    
    true_markers = np.array(aruco_gt)
    #print(f"marker shape: {true_markers.shape[1]}")
    
    true_markers_formatted = np.zeros((2,0))
    
    for marker in true_markers:
        marker = np.array([marker[1], marker[0]])
        marker_formatted = np.array([marker]).T
        print(f"marker: {marker}")
        print(f"marker format: {marker_formatted}")
        true_markers_formatted = np.concatenate((true_markers_formatted, marker_formatted), axis=1)
    
    print(f"markers: {true_markers_formatted}")
    
    operate = OperateModified(args, true_markers_formatted)
    
    ##########
    #REMEMBER TO ENABLE SLAM
    operate.ekf_on = True 
    ##########
    
    # initialise gui
    if args.gui:
        width, height = 640, 640
        canvas = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Semiautomatic Waypoint Selection')
        canvas.fill(WHITE)
        pygame.display.update()
        u0 = width / 2
        v0 = height / 2
        
        RADIUS = 15
        pygame.init()
        font = pygame.font.SysFont("Arial", 15)
    

    # semi-automatic approach for fruit delivery
    while True:
        # enter the waypoints
        # instead of manually enter waypoints, you can get coordinates by clicking on a map, see camera_calibration.py
        x,y = 0.0,0.0
        
        # implement gui
        apple_gt, lemon_gt, person_gt, aruco_gt = parse_map(args.map)
        #print(f"apple_gt: {apple_gt}")
        
        if args.gui:
        # Enter main pygame loop
            choosing_waypoint = True
            while choosing_waypoint:
            
                # Check for any mouse presses
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        pos = pygame.mouse.get_pos()
                        x,y = pix_to_world(pos[0], pos[1])
                        print(f"mouse pos, (u,v): {pos}")
                        print(f"mouse pos, (x,y): {pix_to_world(pos[0], pos[1])}")
                        choosing_waypoint = False
                
                #draw each individual apple
                for apple in apple_gt:
                    pos = world_to_pix(apple[1], apple[0])
                    pygame.draw.circle(canvas, RED, pos, RADIUS)
                    
                for lemon in lemon_gt:
                    pos = world_to_pix(lemon[1], lemon[0])
                    pygame.draw.circle(canvas, YELLOW, pos, RADIUS)
                    
                for person in person_gt:
                    pos = world_to_pix(person[1], person[0])
                    pygame.draw.circle(canvas, BLACK, pos, RADIUS)
                    
                
                for index, aruco in enumerate(aruco_gt):
                    pos = world_to_pix(aruco[1], aruco[0])
                    text_offset = (5, 5)
                    text_pos = tuple(map(lambda i, j: i + j, pos, text_offset))
                    pygame.draw.circle(canvas, BLUE, text_pos, RADIUS)
                    text = font.render(str(index + 1), True, WHITE)
                    canvas.blit(text, pos)
                
                
                
                # Also loop over markers once parse map is edited to work with markers
                pygame.display.update()
            
        else:
         
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
        
        #while not_at_waypoint:
            # check and update slam
            # are we at waypoint?
            # if not, we need to run the set_velocity with our P controller
        slam_to_point(waypoint,operate)
            #robot_pose = drive_to_point(waypoint,robot_pose)
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

        # exit
        operate.pibot.set_velocity([0, 0])
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput == 'N':
            operate.command['output'] = True
            operate.record_data() # Save a copy of the SLAM map to lab_output/slam.txt
            break