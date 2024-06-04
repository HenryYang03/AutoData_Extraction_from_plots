import cv2
import torch
import pytesseract

class BarGraphAnalyzer:
    def __init__(self, model_path, class_names, pytesseract_cmd = None):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path = model_path)
        self.class_names = class_names
        if pytesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = pytesseract_cmd

