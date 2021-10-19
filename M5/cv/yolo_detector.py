# YOLO DETECTOR

import torch

class YOLODetector:

    def __init__(self):
        # Import yolov5 model with our trained weights
        self.model = torch.hub.load('yolov5', 'custom', path='weights/best.pt', source='local')  # or yolov5m, yolov5l, yolov5x, custom
        pass
    
    def infer(self, img):
        self.results = self.model(img)
        return self.results