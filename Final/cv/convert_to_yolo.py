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

import glob
import argparse

# use the machinevision toolbox to get the bounding box of the detected target(s) in an image
def get_bounding_box(image_path):
    image = PIL.Image.open(image_path).resize((640,480), PIL.Image.NEAREST)
    target = Image(image)
    blobs = target.blobs()
    [[u1,u2],[v1,v2]] = blobs[0].bbox # bounding box
    width = abs(u1-u2)
    height = abs(v1-v2)
    center = np.array(blobs[0].centroid).reshape(2,)
    box = [center[0] / float(640), center[1] / float(480), int(width) / float(640), int(height)/ float(480)] # box=[x,y,width,height]
    # plt.imshow(fruit.image)
    # plt.annotate(str(fruit_number), np.array(blobs[0].centroid).reshape(2,))
    # plt.show()
    # assert len(blobs) == 1, "An image should contain only one object of each target type"
    return box

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Fruit delivery")
    parser.add_argument("-d", "--dir", type=str, default='/home/blayke/catkin_ws/src/data_collector/dataset/Synth/data')
    args, _ = parser.parse_known_args()
    
    glob_pattern = os.path.join(args.dir, "*.png")
    label_filenames = glob.glob(glob_pattern)

    for label in label_filenames:
        try:
            base = os.path.basename(label)
            base_split = base.split("_")
            output_name = f"{base_split[0]}_{base_split[1]}.txt"
            class_string = base_split[0]
            if class_string == "apple":
                class_index = 0
            elif class_string == "lemon":
                class_index = 1
            else: #else person
                class_index = 2
                
            box = get_bounding_box(label)
            box_string = "{} {} {} {} {}".format(class_index, box[0], box[1], box[2], box[3])
            
            with open(os.path.join(args.dir, output_name), 'w') as f:
                f.write(box_string)
        
        except:
            #print(label)
            pass
