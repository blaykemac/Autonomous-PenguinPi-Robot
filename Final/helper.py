'''
Set of helper functions used for calculating waypoint errors, translating gui mouse clicks into world coordinates, etc..
'''
# Import dependencies
import numpy as np
import ast
import math


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
def pix_to_world(u,v, u0, v0, width):
    """
    Takes pixel coordinates u,v, the pixel origin  coordinates u0, v0, and the width of the semi-auto GUI window and returns the world coordinates that coorespond to these pixel coordinates.

    """
    x = (u - u0) * 3 / width
    y  = (v0 - v) * 3 / width
    return x, y
    
def world_to_pix(x,y, u0, v0, width):
    """
    Takes world coordinates x,y, the pixel origin coordinates u0, v0, and the width of the semi-auto GUI window to output the pixel coordinates that correspond to the world coordinates.
    """
    u = x * width / 3 + u0
    v = v0 - y * width / 3 
    return u, v

def create_motion_command(control_v, control_omega):
    """
    Takes control signals control_v, control_omega and formats them into a 'motion' command [v, omega] such that v, omega are integers
    """

    if control_v == 0 and control_omega == 0:
        return [0,0]
    if control_v == 0 and not control_omega == 0:
        return [0, int(control_omega / abs(control_omega) * math.ceil(abs(control_omega)))]         
    if not control_v == 0 and control_omega == 0:
        return [int(control_v / abs(control_v) * math.ceil(abs(control_v))), 0]

    return [int(control_v / abs(control_v) * math.ceil(abs(control_v))), int(control_omega / abs(control_omega) * math.ceil(abs(control_omega)))]
    
    
def PControllerOmega(goal_theta, robot_theta, gain):
    """
    Takes the angle towards the goal, goal_theta, the perceived robot theta, robot_theta, the P controller gain K_pw to return the angular velocity control signal, control_signal, as well as the angle error, error_theta
    """
    error_theta = np.arctan2(np.sin(goal_theta - robot_theta), np.cos(goal_theta - robot_theta))
    #error_theta = clamp_angle(goal_theta - theta)
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

 
# read in the object poses, note that the object pose array is [y,x]
def parse_map(fname: str) -> dict:
    with open(fname,'r') as f:
        gt_dict = ast.literal_eval(f.readline())        
        apple_gt, lemon_gt, person_gt, aruco_gt = [], [], [], []
        
        # fill aruco_gt list with 10 blank coordinates
        aruco_gt = [np.array([0,0])] * 10
        #print(f"aruco_gt: {aruco_gt}")

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