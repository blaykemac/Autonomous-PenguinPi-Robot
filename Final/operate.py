##########################################################
# Final fruit delivery control code
##########################################################

# import modules
import matplotlib
matplotlib.use('TkAgg',  force=True)
import matplotlib.pyplot as plt
plt.switch_backend('Agg')


import sys, os
import ast
import numpy as np
import json
import argparse
import pygame
import cv2
import time
import math
import tkinter

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
sys.path.insert(0,"{}/cv/".format(os.getcwd()))
from cv.detector import Detector
from cv.yolo_detector import YOLODetector
from cv.annotate import Annotate
from cv.estimate_pose import *

# import the helper functions
from helper import *
from helper_james import *


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
        self.ekf_on = False
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length = 0.07) # size of the ARUCO markers
        self.request_recover_robot = False

        # initialise default GUI parameters
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
        self.file_output = None
        self.double_reset_confirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        
        # initialise images for default GUI
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.detector_output = np.zeros([240,320], dtype=np.uint8)
        if args.ckpt == "":
            self.detector = None
            self.network_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            self.detector = Detector("model/model.best.pth", use_gpu=False) # DO I NEED TO SET 'USE GPU' TO TRUE IF WE CAN RUN ON MY DESKTOP MACHINE?
            self.yolo_detector = YOLODetector(args.ckpt)
            self.network_vis = np.ones((240, 320,3))* 100
        self.bg = pygame.image.load('pics/gui_mask.jpg')
        self.class_names = ["apple", "lemon", "person"]
        
        if args.load_slam_cv or args.load_slam:
            self.slam_map_loader = dh.InputReader("lab_output")
            self.loaded_taglist, self.loaded_markers, self.loaded_P = self.slam_map_loader.read_slam()
            self.ekf.markers = self.loaded_markers
            self.ekf.taglist = self.loaded_taglist
            
            temp_P = np.zeros((23,23)) # assume we have all 10 markers in slam.txt
            temp_P[:3, :3] = self.ekf.P[:3, :3] 
            temp_P[3:, 3:] = self.loaded_P
            self.ekf.P = temp_P
            
        # then we must also add cv map
        if args.load_slam_cv:
            self.objects_loader = dh.InputReader("lab_output")
            self.object_locations =  self.objects_loader.read_objects()

        else:
            self.object_locations = [[None, None, None], [None, None, None], [None, None, None]] # [[apples_xy], [lemons_xy], [persons_xy]] Initialise as None until we have merged our estimations

        self.estimation_threshold = 1
        self.object_locations_premerge = [[], [], []]
        self.inference_buffer = [[], [], []]
        self.confidence_threshold = 0.7
        self.auto_waypoint_enabled = False
        #self.detections_buffer 
        
        # initialise waypoint
        self.instruction = None
        self.waypoint = None #np.array([0.0, 0.0])
        self.finished_navigating = True # we start off at the waypoint
        self.turning = True
        self.keyboard_overridden = False
        self.prev_distance_error = np.inf
        self.states = {"manual": 0, "begin_automation" : 1, "navigation_turning" : 2, "navigation_forward": 3, "navigation_arrived_waypoint" : 4,
                       "navigation_complete" : 5, "turn_to_lemon": 6, "track_lemon": 7, "push_lemon": 8, "relocate_lemon": 9, "move_backwards": 10}
        self.state = self.states["manual"]
        
        # compute the controller gains, tolerances and parameters
        self.K_pw = 1
        self.angle_tolerance = 0.02 #rads, 0.1 rads ~ 5 degrees
        self.K_pv = 1
        self.dist_tolerance = 0.01 #metres
        
        # initialise rrt parameters
        self.r_true_apple = 0.075
        self.r_true_lemon = 0.06
        self.r_true_person = 0.19
        self.r_true_marker = 0.1
        self.obstacle_padding = 0.08
        self.r_true_apple += self.obstacle_padding
        self.r_true_lemon += self.obstacle_padding
        self.r_true_person += self.obstacle_padding
        self.r_true_marker += self.obstacle_padding
        self.auto_instruction_list = []
        self.route_planning_obstacles = None

        # optionally load the true map instead of using SLAM to draw object locations
        if args.truemap:
            # read true coordinates of map from gazebo backend
            self.apple_gt, self.lemon_gt, self.person_gt, self.aruco_gt = parse_map(args.map)
            #print("Map: \napple = {}, \nlemon = {}, \nperson = {}, \naruco = {}".format(self.apple_gt, self.lemon_gt, self.person_gt, self.aruco_gt))

            # find apple(s) and lemon(s) that need to be moved
            #to_move = compute_dist(self.apple_gt, self.lemon_gt, self.person_gt)
            
        # initialise semiauto GUI
        
        # define some colours for drawing colours on pygame
        self.RED = (255, 0, 0) # apple
        self.YELLOW = (255, 255, 0) #lemon
        self.WHITE = (255,  255, 255) #background
        self.BLUE = (0, 0, 255) #aruco
        self.BLACK = (0, 0, 0) #person
        
        self.gui_clicked = False
        
        pygame.font.init()
        self.default_width, self.default_height = 700, 660
        self.semiauto_gui_width = 660
        
        # update width to account for our semiauto gui
        self.modified_width = self.default_width
        self.modified_height = self.default_height
        
        if args.gui:
            self.u0 = self.semiauto_gui_width / 2 + self.default_width # the coordinate transformation from the overall gui window pixel origin to the center of the custom gui
            self.v0 = self.default_height / 2 # the coorrdinate transformation from the overall gui window pixel origin to the center of the custom gui
            self.RADIUS = 15 # the radius of the objects in pixels for gui
            self.modified_width = self.default_width + self.semiauto_gui_width
        
        self.canvas = pygame.display.set_mode((self.modified_width, self.modified_height))
        pygame.display.set_caption('ECE4078 2021 Lab')
        pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
        self.canvas.fill(self.WHITE)
        
        pygame.init()
        self.font = pygame.font.SysFont("Arial", 15)
        self.TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
        self.TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
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
            
            detection = self.yolo_detector.infer(self.img)
            self.command['inference'] = False
            detections = detection.xywh[0].cpu().numpy() # [[xc, yc, w, h, confidence, class_id], ...., ...]
            self.inference_buffer = [[], [], []] # reinitialise because we have a new detection to go into buffer
            colour_mask = [0] * len(detections) # 1 if we ar e keeping detection
            print(detections)
            print(len(detections))
            detection_coordinates = [None] * len(detections)
            
            for target_index, target in enumerate(detections):
                class_id = int(target[5])
                class_name = self.class_names[class_id]
                box = target[:4]
                robot_pose = self.ekf.robot.state[:3]
                target_world = box_to_world(box, robot_pose, class_name, self.camera_matrix[0][0])
                #target_world_test = box_to_world_mtrx(box, robot_pose, class_name, self.camera_matrix)
                #print(target_world_test)
                confidence = target[4]
                print(f"target_world: {target_world}")
                # gonna need to check if self.premerge flag is true ebefore running 
                distance_from_robot = np.hypot(target_world[0] - self.ekf.robot.state[0], target_world[1] - self.ekf.robot.state[1])
                detection_coordinates[target_index] = target_world
                if distance_from_robot < self.estimation_threshold and confidence > self.confidence_threshold:
                    print("valid")
                    print(f"index, cls {target_index}, {class_id}")
                    print(f"dist: {distance_from_robot}")
                    self.inference_buffer[class_id].append(target_world)
                    colour_mask[target_index] = 1

                    
                else:
                    print("invalid")
                    print(f"index, cls {target_index}, {class_id}")
                    print(f"dist: {distance_from_robot}")
                    
            print(f"mask: {colour_mask}")
            
            annotate = Annotate(detection.imgs, detection.pred, detection.names, colour_mask, detection_coordinates)
            self.network_vis = annotate.get_annotations()

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
        print(self.baseline)
        robot = Robot(self.baseline, self.scale, self.camera_matrix, self.dist_coeffs)
        return EKF(robot)

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False
            
        # save inference with the matching robot pose and detector labels
        #print(f"save infer: {self.command['save_inference']}")
        #print(f"inf buffer: {self.inference_buffer}")
        #print(f"inf buffer != ")
        if self.command['save_inference'] and self.inference_buffer != [[], [], []]:
            for class_index, classes in enumerate(self.inference_buffer):
                if classes != []:
                    for detected_object_coordinates in classes:
                        self.object_locations_premerge[class_index].append(detected_object_coordinates)
            self.notification = "Accepted detection"
       
            """
            if self.file_output is not None:
                #image = cv2.cvtColor(self.file_output[0], cv2.COLOR_RGB2BGR)
                self.pred_fname = self.output.write_image(self.file_output[0],
                                                        self.file_output[1])
                self.notification = f'Prediction is saved to {operate.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
                """
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
        if self.args.gui:
            # if reading object coordinates from TRUEMAP instead of through SLAM for semiauto GUI
            if self.args.truemap:
                pygame.draw.rect(self.canvas, self.WHITE, pygame.Rect(self.default_width, 0, self.semiauto_gui_width, self.default_height))
                pos = world_to_pix(self.ekf.robot.state[0][0], self.ekf.robot.state[1][0], self.u0, self.v0, self.semiauto_gui_width)
                pygame.draw.circle(self.canvas, (40, 40, 40), pos, self.RADIUS*1.5)
            
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
                pygame.draw.rect(self.canvas, self.WHITE, pygame.Rect(self.default_width, 0, self.semiauto_gui_width, self.default_height))
                pos = world_to_pix(self.ekf.robot.state[0][0], self.ekf.robot.state[1][0], self.u0, self.v0, self.semiauto_gui_width)
                pygame.draw.circle(self.canvas, (40, 40, 40), pos, self.RADIUS*1.5)
            
                for apple in self.object_locations[0]:
                    if apple is not None:
                        pos = world_to_pix(apple[0], apple[1], self.u0, self.v0, self.semiauto_gui_width)
                        pygame.draw.circle(self.canvas, self.RED, pos, self.RADIUS)
                    
                for lemon in self.object_locations[1]:
                    if lemon is not None:
                        pos = world_to_pix(lemon[0], lemon[1], self.u0, self.v0, self.semiauto_gui_width)
                        pygame.draw.circle(self.canvas, self.YELLOW, pos, self.RADIUS)
                    
                for person in self.object_locations[2]:
                    if person is not None:
                        pos = world_to_pix(person[0], person[1], self.u0, self.v0, self.semiauto_gui_width)
                        pygame.draw.circle(self.canvas, self.BLACK, pos, self.RADIUS)
                # for index, aruco in enumerate(self.ekf.markers)
                for index in range(self.ekf.markers.size//2):
                    aruco = self.ekf.markers[:, index]
                    pos = world_to_pix(aruco[0], aruco[1], self.u0, self.v0, self.semiauto_gui_width)
                    text_offset = (5, 5)
                    text_pos = tuple(map(lambda i, j: i + j, pos, text_offset))
                    pygame.draw.circle(self.canvas, self.BLUE, text_pos, self.RADIUS)
                    text = self.font.render(str(self.ekf.taglist[index]), True, self.WHITE)
                    self.canvas.blit(text, pos)


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
                self.state = self.states["manual"]
                
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'][0] = max(self.command['motion'][0]-1, -1)
                self.state = self.states["manual"]
                
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'][1] = min(self.command['motion'][1]+1, 1)
                self.state = self.states["manual"]
                
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'][1] = max(self.command['motion'][1]-1, -1)
                self.state = self.states["manual"]
                
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
                self.state = self.states["manual"]
                
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
                
            # start running the fuill auto fruit delivery code
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                self.state = self.states["begin_automation"]

            # merge estimations and save to targets.txt
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                # merge the estimations of the targets so that there are at most 3 estimations of each target type
                self.target_est_dict, self.object_locations = merge_estimations(self.object_locations_premerge)
                     
                # save target pose estimations
                with open('lab_output/targets.txt', 'w') as fo:
                    json.dump(self.target_est_dict, fo)
                    self.notification = "Saved object locations to targets.txt"
                
                print('Estimations saved!')
                
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
                
            # check for mouse input --- CURRENTLY CHECKING ALL MOUSE BUTTONS _ CHANGE TO LEFT AND RIGHT
            if event.type == pygame.MOUSEBUTTONDOWN and self.args.gui and event.button == 3:
                dist_reverse = 0.1
                x = self.ekf.robot.state[0][0]
                y = self.ekf.robot.state[1][0]
                angle = self.ekf.robot.state[2][0]
                collinear_point = np.array([x, y]) - np.array([np.cos(angle), np.sin(angle)])
                self.waypoint = go_x_dist_in_dir(np.array([x, y]), collinear_point, dist_reverse)
                self.state = self.states["move_backwards"]

            if event.type == pygame.MOUSEBUTTONDOWN and self.args.gui and event.button == 1:
                pos = pygame.mouse.get_pos()
                
                # flag that tells us if the gui has been clicked (used to help auto navigation)
                self.gui_clicked = True

                if pos[0] >= self.default_width:
                    x,y = pix_to_world(pos[0], pos[1], self.u0, self.v0, self.semiauto_gui_width)
                    self.waypoint = np.array([x, y])
                    self.state = self.states["navigation_turning"]
                
        if self.quit:
            pygame.quit()
            sys.exit()
    
    def state_transition(self):
        
        if self.state == self.states["manual"]:
            pass
    
        elif self.state == self.states["begin_automation"]:
            self.auto_instruction_list = self.generate_instruction_list()
            self.auto_instruction_list.reverse()
            if len(self.auto_instruction_list) > 0:
                    self.auto_instruction_list.pop()
                    self.state = self.states["navigation_arrived_waypoint"]

            else:
                print("Failed to generate waypoint list")
            
        elif self.state == self.states["navigation_turning"]:
            error_theta = get_angle_robot_to_goal(self.waypoint, self.ekf.robot.state)
            if abs(error_theta) > self.angle_tolerance:
                control_v = 0 # We are only turning here
                control_omega, _ = PControllerOmegaDynamic(self.waypoint, self.ekf.robot.state, self.K_pw)
                self.command['motion'] = create_motion_command(control_v, control_omega)
            else:
                self.command['motion'] = [0,0]
                self.state = self.states["navigation_forward"]
            
        elif self.state == self.states["navigation_forward"]:
            error_dist = get_distance_robot_to_goal(self.waypoint, self.ekf.robot.state)
            #if abs(error_dist) < abs(self.prev_distance_error):
            if abs(error_dist) > abs(self.dist_tolerance) and abs(error_dist) < abs(self.prev_distance_error):
            
                self.prev_distance_error = error_dist
                control_omega = 0 # only driving straight
                control_v, _ = PControllerV(self.waypoint, self.ekf.robot.state, self.K_pv)
                self.command['motion'] = create_motion_command(control_v, control_omega)
            else:
                self.prev_distance_error = np.inf
                self.command['motion'] = [0,0]

                self.state = self.states["navigation_arrived_waypoint"]
                
        elif self.state == self.states["navigation_arrived_waypoint"]:
            # we have moved blindly. go back to waiting for new command
            if self.args.gui:
                self.state = self.states["manual"]
            else:
                if len(self.auto_instruction_list) > 0:
                    self.instruction = self.auto_instruction_list.pop()
                    self.waypoint = self.instruction.point
                    """
                    if self.instruction.tag == 0: # regular navigation waypoint
                        self.state = self.states["navigation_turning"]
                    elif self.instruction.tag == 1: # lemon moving waypoint
                        self.state = self.states["turn_to_lemon"]
                    """
                    self.state = self.states["navigation_turning"]
                else:
                    self.state = self.states["navigation_complete"]

        # needed something to do just a turn before tracking the lemon, this is just copy paste nav turning state
        elif self.state == self.states["turn_to_lemon"]:
            error_theta = get_angle_robot_to_goal(self.waypoint, self.ekf.robot.state)
            if abs(error_theta) > self.angle_tolerance:
                control_v = 0 # We are only turning here
                control_omega, _ = PControllerOmegaDynamic(self.waypoint, self.ekf.robot.state, self.K_pw)
                self.command['motion'] = create_motion_command(control_v, control_omega)
            else:
                self.command['motion'] = [0, 0]
                self.state = self.states["track_lemon"]


        elif self.state == self.states["track_lemon"]:
            # once lemon is located, record lemons position relative to current pos (probably make this a state)

            potential_targets = self.detect_class("lemon")

            if len(potential_targets) == 0:
                # should've found a lemon, just move on for now
                self.state = self.states['navigation_arrived_waypoint']
            else:
                # found lemon(s), find one which corresponds to the target lemon
                current_best_lemon = self.find_closest_point_match(self.instruction.target.point, potential_targets)

                # update object map to contain new lemon position
                self.update_object_map(self.instruction.target.point, current_best_lemon, obj_class)
                self.update_route_planning_obs(self.instruction.target.point, current_best_lemon)

                # calculate new waypoint(s) to push lemon parallel to previous trajectory
                prev_traj = (self.instruction.point, self.auto_instruction_list[-1].point)  # src, dest
                new_traj = calculate_new_traj_parallel(prev_traj, intended_lemon_target[0:2], current_best_lemon)
                # if new trajectory is very similar to previous one, traj is probably correct, may now push lemon
                matched = compare_traj(prev_traj, new_traj)
                if matched:
                    self.instruction = Instruction(new_traj[1], 1, target=current_best_lemon)
                    self.waypoint = self.instruction.point
                    self.state = self.states["push_lemon"]
                # else, make a navigation waypoint to new aligned position and repeat process
                else:
                    # add nav waypoint and track lemon waypoint to instruction list
                    self.auto_instruction_list.append(Instruction(new_traj[1], 1)) # append the lemon align instruction
                    self.auto_instruction_list.append(Instruction(new_traj[0], 0)) # append the nav instruction BEFORE since these get popped
                    self.state = self.states["navigation_arrived_waypoint"]


        elif self.state == self.states["push_lemon"]:
            # in position now, time to actually move the lemon
            error_dist = get_distance_robot_to_goal(self.waypoint, self.ekf.robot.state)
            # if abs(error_dist) < abs(self.prev_distance_error):
            if abs(error_dist) > abs(self.dist_tolerance) and abs(error_dist) < abs(self.prev_distance_error):
                self.prev_distance_error = error_dist
                control_omega = 0  # only driving straight
                control_v, _ = PControllerV(self.waypoint, self.ekf.robot.state, self.K_pv)
                self.command['motion'] = create_motion_command(control_v, control_omega)
            else:
                self.prev_distance_error = np.inf
                self.command['motion'] = [0, 0]
                self.state = self.states["move_backwards"]
                self.waypoint = go_x_dist_in_dir(self.waypoint, self.instruction.target, 0.1)


        elif self.state == self.states["move_backwards"]:
            # modify this to let robot go backwards??????
            # in position now, time to actually move the lemon
            error_dist = get_distance_robot_to_goal(self.waypoint, self.ekf.robot.state)
            # if abs(error_dist) < abs(self.prev_distance_error):
            if abs(error_dist) > abs(self.dist_tolerance) and abs(error_dist) < abs(self.prev_distance_error):
                self.prev_distance_error = error_dist
                control_omega = 0  # only driving straight
                control_v, _ = PControllerV(self.waypoint, self.ekf.robot.state, self.K_pv)
                self.command['motion'] = create_motion_command(-control_v, control_omega) # negative here, should work
            else:
                self.prev_distance_error = np.inf
                self.command['motion'] = [0, 0]
                if self.args.gui:
                    self.state = self.states["manual"]
                else:
                    self.state = self.states["relocate_lemon"]


        elif self.state == self.states["relocate_lemon"]:
            potential_targets = self.detect_class(class_name)
            if len(potential_targets) == 0:
                # should've found a lemon, just move on for now
                self.state = self.states['navigation_arrived_waypoint']
            else:
                # found lemon(s), find one which corresponds to the target lemon
                current_best_lemon = self.find_closest_point_match(self.instruction.target.point, potential_targets)

                # update object map to contain new lemon position
                self.update_object_map(self.instruction.target.point, current_best_lemon, obj_class)
                self.update_route_planning_obs(self.instruction.target.point, current_best_lemon)

                # check if pushed lemon is obstructing the next path
                if collision_between_points(self.instruction.point, self.auto_instruction_list[-1].point, self.route_planning_obstacles):
                    # if so, recompute path to the next objective
                    next_goal_idx = None
                    for idx in range(len(self.auto_instruction_list)):
                        if self.auto_instruction_list[-(i+1)].tag != 0:
                            next_goal_idx = idx
                    # generate path to
                    if next_goal_idx is None:
                        # no next goal, just stop
                        self.state = self.states['navigation_arrived_waypoint']
                    else:
                        target_dest = self.auto_instruction_list[-(next_goal_idx + 1) + 1].point
                        rrt = RRT(start=self.instruction.point, goal=target_dest, width=1.4, height=1.4, obstacle_list=self.route_planning_obstacles, expand_dis=0.2, path_resolution=0.04)
                        route = rrt.planning()
                        #route.reverse() # ???
                        new_nav_instr = [Instruction(n, 0) for n in route]
                        self.auto_instruction_list = self.auto_instruction_list[:-(next_goal_idx + 1)] # slice off old instructions
                        self.auto_instruction_list += new_nav_instr
                        self.state = self.states['navigation_arrived_waypoint']


    def update_object_map(self, obj_to_update, updated_obj, obj_class):
        for i, obj in enumerate(self.object_locations[self.class_names.index(obj_class)]):
            if np.array_equal(obj, obj_to_update):
                self.object_locations[self.class_names.index(obj_class)][i] = updated_obj


    def update_route_planning_obs(self, obj_to_update, updated_obj):
        for i, obj in enumerate(self.route_planning_obstacles):
            if np.array_equal(obj.center, obj_to_update):
                self.route_planning_obstacles[i] = updated_obj


    def detect_class(self, class_name_tgt):
        detection = self.yolo_detector.infer(self.img)
        detections = detection.xywh[0].cpu().numpy()
        robot_pose = self.ekf.robot.state[:3]
        potential_targets = []

        for target in detections:
            class_id = int(target[5])
            class_name = self.class_names[class_id]
            if class_name != class_name_tgt:
                continue
            box = target[:4]
            target_world = box_to_world(box, robot_pose, class_name, self.camera_matrix[0][0])
            confidence = target[4]
            distance_from_robot = np.hypot(target_world[0] - self.ekf.robot.state[0],
                                           target_world[1] - self.ekf.robot.state[1])
            if distance_from_robot < self.estimation_threshold and confidence > self.confidence_threshold:
                print("valid")
                potential_targets.append((target_world, distance_from_robot))

        return potential_targets

                
    def generate_instruction_list(self):
    
        start_point = np.array([self.ekf.robot.state[0][0], self.ekf.robot.state[1][0]])
        self.route_planning_obstacles = []
        for entry in self.object_locations[0]:
            self.route_planning_obstacles.append(CircleT(entry[0], entry[1], self.r_true_apple, 0))

        for entry in self.object_locations[1]:
            self.route_planning_obstacles.append(CircleT(entry[0], entry[1], self.r_true_lemon, 1))
    
        for entry in self.object_locations[2]:
            self.route_planning_obstacles.append(CircleT(entry[0], entry[1], self.r_true_person, 2))
    
        for i in range(int(self.ekf.markers.size/2)):
            self.route_planning_obstacles.append(CircleT(self.ekf.markers[0, i], self.ekf.markers[1, i], self.r_true_marker, 3))
                    

        lemon_not_done = objects_not_done(self.object_locations[0], self.object_locations[1], self.object_locations[2])
        print("Objects not done")
        print(f"Lemons- {lemon_not_done}")

        # TEMP
        # lemon_not_done = list(self.object_locations[1])
        apple_not_done = list(self.object_locations[0])
        fruit_not_done = lemon_not_done + apple_not_done

        instructions, pathlength, log = generate_fruit_path(0, 0, fruit_not_done, self.route_planning_obstacles, start_point, 20)

        print("instructions generated")
        print(log)
        animate_path_x(np.array([n.point for n in instructions]), (-1.5, 1.5), (-1.5, 1.5), self.route_planning_obstacles)
        # plt.show()
        plt.savefig("rrt.png")


        
        return instructions
        #rrt = RRT(start=start_point, goal=goal_point, width=1.4, height=1.4, obstacle_list=all_obstacles, expand_dis=0.2, path_resolution=0.04)
        
        """
        rrt = RRT(start=start_point, goal=goal_point, width=1.4, height=1.4, obstacle_list=all_obstacles, expand_dis=0.4, path_resolution=0.04)
        
        # generate plans and try choose the best one based off-
        # number of nodes and total distance travelled.
        plans = []
        self.MAX_PLANS = 20
        for plans_index in range(self.MAX_PLANS):
            timeout_counter = 1
            # only attempt to make a plan a finite amount of time
            while timeout_counter <= timeout:
                timeout_counter += 1
                plan = rrt.planning()
                if plan:
                    plans.append(plan)
                    break
        plan_lengths = [len(plan) for plan in plans]
        min_plans_length = min(plan_lengths)
        for index, plan_length in enumerate(plan_lengths):
            if plan_length == min_plans_length:
                return plans[index]

        print([len(plan) for plan in plans])
        print(len(optimal_plan))
        return optimal_plan
        
        """
            
        