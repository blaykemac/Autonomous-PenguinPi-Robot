# annotate

import sys
import os
from pathlib import Path

# add utils to path to avoid yolo errors
current_file_directory = os.path.dirname(os.path.abspath(__file__))
path = Path(current_file_directory)
parent_path = path.parent.absolute()
#sys.path.append(parent_path)
sys.path.append("cv/yolov5/")

# only import after adding path 
from yolov5.utils.plots import Annotator, colors
from PIL import Image, ImageDraw, ImageFont
import cv2

class Annotate:
    def __init__(self, imgs, pred, names, mask, coordinates):
        self.imgs = imgs
        self.pred = pred
        self.names = names
        self.annotator = None
        self.mask = mask
        self.mask.reverse() # reverse to match this code for loop
        self.coordinates = coordinates
        
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            if pred.shape[0]:
                self.annotator = Annotator(im, example=str(self.names))
                for detection_index, (*box, conf, cls) in enumerate(reversed(pred)):  # xyxy, confidence, class
                    if self.coordinates == []:
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                    else:
                        label = f'{self.names[int(cls)]} {conf:.2f} x:{self.coordinates[detection_index][0]:.2f}, y: {self.coordinates[detection_index][1]:.2f}'
                    self.annotator.box_label(box, label, color=colors(14 if not self.mask[detection_index] else cls))

    def get_annotations(self):
        if self.annotator is not None:
            img_rgb = self.annotator.result()
            #img_rgb = cv2.cvtColor(self.annotator.result(), cv2.COLOR_BGR2RGB)
        else:
            #img_rgb = cv2.cvtColor(self.imgs[0], cv2.COLOR_BGR2RGB)
            img_rgb = self.imgs[0]
        return img_rgb
   