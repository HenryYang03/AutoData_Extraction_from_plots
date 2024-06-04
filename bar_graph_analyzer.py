import cv2
import torch
import pytesseract

class BarGraphAnalyzer:
    def __init__(self, model_path, class_names, pytesseract_cmd = None):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path = model_path)
        self.class_names = class_names
        if pytesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = pytesseract_cmd

    def analyze_image(self, image_path):
        # Load the image
        image = cv2.imread(image_path)

        # Run YOLO model on the image
        results = self.model(image)

        # Extract the detections
        detections = results.xyxy[0].numpy()  # (x1, y1, x2, y2, conf, cls)

        # Filter and group the detections
        bar_groups = self.group_bars(detections, image)

        # Calculate true heights for each group
        results = self.calculate_heights(bar_groups, image)

        return results