##########################################################
# Final fruit delivery control code
##########################################################


# import modules
import sys, os
import ast
import numpy as np
import json
import argparse
import pygame
import cv2
import time
import math


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


# import CV components
sys.path.insert(0,"{}/network/".format(os.getcwd()))
sys.path.insert(0,"{}/network/scripts".format(os.getcwd()))
from network.scripts.detector import Detector

# import the helper functions
from helper import *

class Operate:
    def __init__(self, args):
        self.args = args
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
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
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
        self.double_reset_confirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        
        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.detector_output = np.zeros([240,320], dtype=np.uint8)
        if args.ckpt == "":
            self.detector = None
            self.network_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector(args.ckpt, use_gpu=False) # DO I NEED TO SET 'USE GPU' TO TRUE IF WE CAN RUN ON MY DESKTOP MACHINE?
            self.network_vis = np.ones((240, 320,3))* 100
        self.bg = pygame.image.load('pics/gui_mask.jpg')
        
         # define some colours for drawing colours on pygame
        self.RED = (255, 0, 0) # apple
        self.YELLOW = (255, 255, 0) #lemon
        self.WHITE = (255,  255, 255) #background
        self.BLUE = (0, 0, 255) #aruco
        self.BLACK = (0, 0, 0) #person
        
        # initialise waypoint
        self.waypoint = np.array([0.0, 0.0])
        self.finished_navigating = True # we start off at the waypoint
        self.turning = True
        self.keyboard_overridden = False
        
        # compute the controller gains, tolerances and parameters
        self.K_pw = 1
        self.angle_tolerance = 0.1 #rads, 0.1 rads ~ 5 degrees
        self.K_pv = 1
        self.dist_tolerance = 0.1 #metres
        
        # optionally load the true map instead of using SLAM to draw object locations
        if args.truemap:
            # read true coordinates of map from gazebo backend
            self.apple_gt, self.lemon_gt, self.person_gt, self.aruco_gt = parse_map(args.map)
            print("Map: \napple = {}, \nlemon = {}, \nperson = {}, \naruco = {}".format(self.apple_gt, self.lemon_gt, self.person_gt, self.aruco_gt))

            # find apple(s) and lemon(s) that need to be moved
            to_move = compute_dist(self.apple_gt, self.lemon_gt, self.person_gt)
            print("Fruits to be moved: ", to_move)
            
        # initialise semiauto GUI
        
        pygame.font.init()
        self.default_width, self.default_height = 700, 660
        self.semiauto_gui_width = 660
        
        # update width to account for our semiauto gui
        self.modified_width = self.default_width
        self.modified_height = self.default_height
        
        if not args.nogui:
            self.u0 = self.semiauto_gui_width / 2 + self.default_width # the coordinate transformation from the overall gui window pixel origin to the center of the custom gui
            self.v0 = self.default_height / 2 # the coorrdinate transformation from the overall gui window pixel origin to the center of the custom gui
            self.RADIUS = 15 # the radius of the objects in pixels for gui
            self.modified_width = self.default_width + self.semiauto_gui_width
        
        self.canvas = pygame.display.set_mode((self.modified_width, self.modified_height))
        pygame.display.set_caption('ECE4078 2021 Lab')
        pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
        self.canvas.fill(self.WHITE)
        
        
        """ These images are used in menu that waits for button press to start, uncomment if adding that menu back
        self.pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                         pygame.image.load('pics/8bit/pibot2.png'),
                         pygame.image.load('pics/8bit/pibot3.png'),
                        pygame.image.load('pics/8bit/pibot4.png'),
                         pygame.image.load('pics/8bit/pibot5.png')]
         """
        pygame.init()
        self.font = pygame.font.SysFont("Arial", 15)
        self.TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
        self.TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
        #splash = pygame.image.load('pics/loading.png') # uncomment this for the menu that waits for button press
        pygame.display.update()
            
    # wheel control
    def control(self):       
        if self.args.play_data:
            lv, rv = self.pibot.set_velocity()            
        else:
            lv, rv = self.pibot.set_velocity(
                self.command['motion'])
        if not self.data is None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        drive_meas = measure.Drive(lv, rv, dt)
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
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            self.detector_output, self.network_vis = self.detector.detect_single_image(self.img)
            self.command['inference'] = False
            self.file_output = (self.detector_output, self.ekf)
            self.notification = f'{len(np.unique(self.detector_output))-1} target type(s) detected'

    # save raw images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        self.camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        self.dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        self.scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            self.scale /= 2
        fileB = "{}baseline.txt".format(datadir)  
        self.baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(self.baseline, self.scale, self.camera_matrix, self.dist_coeffs)
        return EKF(robot)

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False
        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:
            if self.file_output is not None:
                #image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)
                self.pred_fname = self.output.write_image(self.file_output[0],
                                                        self.file_output[1])
                self.notification = f'Prediction is saved to {operate.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False

    # paint the GUI            
    def draw(self):
        self.canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480+v_pad),
            not_pause = self.ekf_on)
        self.canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(self.canvas, robot_view, 
                                position=(h_pad, v_pad)
                                )

        # for target detector (M3)
        detector_view = cv2.resize(self.network_vis,
                                   (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(self.canvas, detector_view, 
                                position=(h_pad, 240+2*v_pad)
                                )

        # self.canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(self.canvas, caption='SLAM', position=(2*h_pad+320, v_pad), font = self.TITLE_FONT)
        self.put_caption(self.canvas, caption='Detector',
                         position=(h_pad, 240+2*v_pad), font = self.TITLE_FONT)
        self.put_caption(self.canvas, caption='PiBot Cam', position=(h_pad, v_pad), font = self.TITLE_FONT)

        notification = self.TEXT_FONT.render(self.notification,
                                          False, text_colour)
        self.canvas.blit(notification, (h_pad+10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = self.TEXT_FONT.render(time_remain, False, (50, 50, 50))
        self.canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
        #return canvas #not sure why this was being returned, this is useless for us though
        
        # if using semiauto GUI, paint that too
        if not self.args.nogui:
            # if reading object coordinates from TRUEMAP instead of through SLAM for semiauto GUI
            if self.args.truemap:
                for apple in self.apple_gt:
                    pos = world_to_pix(apple[1], apple[0], self.u0, self.v0, self.semiauto_gui_width)
                    pygame.draw.circle(self.canvas, self.RED, pos, self.RADIUS)
                    
                for lemon in self.lemon_gt:
                    pos = world_to_pix(lemon[1], lemon[0], self.u0, self.v0, self.semiauto_gui_width)
                    pygame.draw.circle(self.canvas, self.YELLOW, pos, self.RADIUS)
                    
                for person in self.person_gt:
                    pos = world_to_pix(person[1], person[0], self.u0, self.v0, self.semiauto_gui_width)
                    pygame.draw.circle(self.canvas, self.BLACK, pos, self.RADIUS)
                
                for index, aruco in enumerate(self.aruco_gt):
                    pos = world_to_pix(aruco[1], aruco[0], self.u0, self.v0, self.semiauto_gui_width)
                    text_offset = (5, 5)
                    text_pos = tuple(map(lambda i, j: i + j, pos, text_offset))
                    pygame.draw.circle(self.canvas, self.BLUE, text_pos, self.RADIUS)
                    text = self.font.render(str(index + 1), True, self.WHITE)
                    self.canvas.blit(text, pos)
                    
            # otherwise use SLAM locations of objects to draw in semiauto GUI
            else:
                # TO DO - IMPLEMENT THIS CASE
                pass

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, font, text_colour=(200, 200, 200)):
        caption_surface = font.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))

    # keyboard and mouse teleoperation        
    def update_input(self):
        for event in pygame.event.get():
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'][0] = min(self.command['motion'][0]+1, 1)
                self.keyboard_overridden = True
                
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'][0] = max(self.command['motion'][0]-1, -1)
                self.keyboard_overridden = True
                
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'][1] = min(self.command['motion'][1]+1, 1)
                self.keyboard_overridden = True
                
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'][1] = max(self.command['motion'][1]-1, -1)
                self.keyboard_overridden = True
                
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
                self.keyboard_overridden = True
                
            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
                
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
                
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_confirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_confirm +=1
                elif self.double_reset_confirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_confirm = 0
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
                
            # check for mouse input
            if event.type == pygame.MOUSEBUTTONDOWN and not self.args.nogui:
                pos = pygame.mouse.get_pos()
                
                # then we actually clicked in the semiauto window
                if pos[0] >= self.default_width:
                    x,y = pix_to_world(pos[0], pos[1], self.u0, self.v0, self.semiauto_gui_width)
                    self.waypoint = np.array([x,y])
                    self.keyboard_overridden = False
                    self.finished_navigating = False
                    self.turning = True
                    
            if self.keyboard_overridden:
                self.finished_navigating = True
                self.turning = True
                
        if self.quit:
            #self.pibot.set_velocity([0, 0])
            pygame.quit()
            sys.exit()
    
    def navigate_to_waypoint(self):

        # Check if we are still navigating to waypoint or whether we are awaiting a new waypoint
        if not self.finished_navigating:
        
            # Compute both angular and radial error
            error_theta = get_angle_robot_to_goal(self.waypoint, self.ekf.robot.state)
            error_dist = get_distance_robot_to_goal(self.waypoint, self.ekf.robot.state)
        
            # If error in angle is too large, we need to generate a new omega command to correct angle
            if abs(error_theta) > self.angle_tolerance and self.turning:
                control_v = 0 # We are only turning here
                control_omega, _ = PControllerOmegaDynamic(self.waypoint, self.ekf.robot.state, self.K_pw)
                self.command['motion'] = create_motion_command(control_v, control_omega)
                
            # If error in distance is too large, we need to keep driving straight to get to waypoint, else we are done navigating
            elif abs(error_dist) > self.dist_tolerance:
                self.turning = False
                control_omega = 0 # only driving straight
                control_v, _ = PControllerV(self.waypoint, self.ekf.robot.state, self.K_pv)
                self.command['motion'] = create_motion_command(control_v, control_omega)
                
            # Done navigating, awaiting new waypoint now
            else:
                self.command['motion'] = [0,0]
                self.finished_navigating = True
                self.turning = True
                
    def automate_waypoint(self):
        """ Add code that genrates waypoint automatically
    
        """
        if self.args.auto:
            #self.waypoint = np.array([x, y])
            self.turning = True
            self.finished_navigating = False
            pass
                

"""
    def blind_drive(self):
        
        #Force the robot to turn and drive to waypoint blindly.
        #NOTE TO USE THIS WITH SLAM, WE WOULD NEED TO RETURN DRIVE_MEAS SIMILAR TO HOW OPERATE.CONTROL WORKS
        

        wheel_vel = 30 # tick per second 
        turning_vel = 5
        
        # turn towards the waypoint
        angle_to_waypoint = np.arctan2((self.waypoint[1] - self.ekf.robot.state[1]) , (self.waypoint[0] - self.ekf.robot.state[0]))
        delta_angle = np.arctan2(np.sin(angle_to_waypoint - self.ekf.robot.state[2]), np.cos(angle_to_waypoint - self.ekf.robot.state[2]))
        modified_wheel_vel = turning_vel
        if delta_angle < 0:
            modified_wheel_vel *= -1
        distance_turning = self.baseline / 2 * abs(delta_angle)
        turn_time = distance_turning / (self.scale * turning_vel)
        print("Turning for {:.2f} seconds".format(turn_time))
        ppi.set_velocity([0, 1], turning_tick=modified_wheel_vel, time=turn_time)
        
        # after turning, drive straight to the waypoint
        distance_straight = np.sqrt((self.waypoint[0] - self.ekf.robot.state[0])**2 + (self.waypoint[1] - self.ekf.robot.state[1])**2)
        drive_time = distance_straight / (self.scale * wheel_vel) # replace with your calculation
        print("Driving for {:.2f} seconds".format(drive_time))
        ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)

        # update the robot pose [x,y,theta]
        robot_pose = [self.waypoint[0], self.waypoint[1], angle_to_waypoint] 
        ####################################################
        return robot_pose
"""