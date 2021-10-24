# YOLO DETECTOR

import torch
import os

class YOLODetector:

    def __init__(self, ckpt):
        # Import yolov5 model with our trained weights
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        yolo_path = os.path.join(current_file_directory, "yolov5")
        #weights_path = os.path.join(current_file_directory, "weights/best.pt")
        weights_path = os.path.join(current_file_directory, ckpt)
        self.model = torch.hub.load(yolo_path, 'custom', path=weights_path, source='local')  # or yolov5m, yolov5l, yolov5x, custom
        pass
    
    def infer(self, img):
        self.results = self.model(img)
        return self.results