# measure performance of delivery
import ast
import numpy as np
import json

# read in the maps
def parse_map(fname: str) -> dict:
    with open(fname,'r') as f:
        gt_dict = ast.literal_eval(f.readline())        
        apple_gt, lemon_gt, person_gt, marker_gt = [], [], [], []

        # remove unique id of targets of the same type 
        for key in gt_dict:
            if key.startswith('apple'):
                apple_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
            elif key.startswith('lemon'):
                lemon_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
            elif key.startswith('person'):
                person_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
            elif key.startswith('aruco'):
                marker_gt.append(np.array(list(gt_dict[key].values()), dtype=float))
    
    # if more than 3 estimations are given for a target type, only the first 3 estimations will be used
    if len(apple_gt) > 3:
        apple_gt = apple_gt[0:3]
    if len(lemon_gt) > 3:
        lemon_gt = lemon_gt[0:3]
    if len(person_gt) > 3:
        person_egt = person_gt[0:3]
    
    return apple_gt, lemon_gt, person_gt, marker_gt


# compute the Euclidean distance between each target's original and current location
# for person models and marker blocks, if the distance is further than 0.1 we consider them moved and penalty will apply
def compute_dist_moved(gt_list, est_list):
    gt_list = gt_list
    est_list = est_list
    dist_list = []
    dist = []
    for gt in gt_list:
        # find the closest current location for each target
        for est in est_list:
            dist.append(np.linalg.norm(gt-est)) # compute Euclidean distance
        dist.sort()
        if dist[0] > 0.1:
            print('Suspect collision during delivery (object original [y x]):', gt)
        dist_list.append(dist[0]) # distance between the target and its closest estimation
        dist = []
    return dist_list

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
                print('All apples too far from Person (current [y x]):', person)
            except:
                pass
        # find lemons that are too close
        for lemon in lemon_list:
            if abs(np.linalg.norm(lemon-person)) < 0.5:
                to_move['person_'+str(i)]['lemon_'+str(k)] = lemon
                to_move['person_'+str(i)]['dist_'+str(k)] = abs(np.linalg.norm(lemon-person))
                print('There are lemons too close to Person (current [y x]):', person)
            k = k+1
        i = i+1
    return to_move

# main program
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Matching the maps before and after delivery")
    parser.add_argument("--original", type=str, default='/home/blayke/catkin_ws/src/penguinpi_gazebo/TRUEMAP.txt')
    parser.add_argument("--current", type=str, default='/home/blayke/catkin_ws/src/penguinpi_gazebo/layout_after_delivery.txt')
    args, _ = parser.parse_known_args()

    # read in maps before and after delivery
    apple_gt, lemon_gt, person_gt, marker_gt = parse_map(args.original)
    apple_est, lemon_est, person_est, marker_est = parse_map(args.current)
    
    # find person without apple and person with lemons too close
    print("Delivery outcomes:")
    compute_dist(apple_est, lemon_est, person_est)
    
    # find person models that moved
    print("\nPerson models that the robot may have collided with during delivery:")
    compute_dist_moved(person_gt, person_est)
    # find marker blocks that moved
    print("\nMarker blocks that the robot may have collided with during delivery:")
    compute_dist_moved(marker_gt, marker_est)
