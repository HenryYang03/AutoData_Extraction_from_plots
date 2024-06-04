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

    def filter_detections(self, detections):
        bars = [d for d in detections if self.class_names[int(d[5])] == 'bar']
        yaxes = [d for d in detections if self.class_names[int(d[5])] == 'yaxis']
        xaxes = [d for d in detections if self.class_names[int(d[5])] == 'xaxis']
        origins = [d for d in detections if self.class_names[int(d[5])] == 'origin']
        ymaxes = [d for d in detections if self.class_names[int(d[5])] == 'ymax']
        labels = [d for d in detections if self.class_names[int(d[5])] == 'label']
        return bars, yaxes, xaxes, origins, ymaxes, labels

    def group_bars(self, detections, image):
        bars, yaxes, xaxes, origins, ymaxes, labels = self.filter_detections(detections)

        if not yaxes or not xaxes:
            raise ValueError("No y-axis or x-axis detected in the image.")

        bar_groups = {}
        for yaxis in yaxes:
            yaxis_mid_x = (yaxis[0] + yaxis[2]) / 2
            corresponding_xaxis = self.find_closest_xaxis(yaxis_mid_x, xaxes)
            corresponding_origin = self.find_closest_label(yaxis_mid_x, origins)
            corresponding_ymax = self.find_closest_label(yaxis_mid_x, ymaxes)
            corresponding_label = self.find_closest_label(yaxis_mid_x, labels)

            if corresponding_xaxis is not None and corresponding_origin is not None and corresponding_ymax is not None and corresponding_label is not None:
                label_text = self.extract_text_from_image(image, corresponding_label, rotate=True)
                group_bars = self.find_group_bars(yaxis, corresponding_xaxis, bars)
                bar_groups[label_text] = (group_bars, yaxis, corresponding_xaxis, corresponding_origin, corresponding_ymax)

        return bar_groups