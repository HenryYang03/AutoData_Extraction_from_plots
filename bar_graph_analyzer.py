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

    def find_closest_xaxis(self, yaxis_mid_x, xaxes):
        min_distance = float('inf')
        corresponding_xaxis = None
        for xaxis in xaxes:
            xaxis_mid_x = (xaxis[0] + xaxis[2]) / 2
            distance = abs(yaxis_mid_x - xaxis_mid_x)
            if distance < min_distance:
                min_distance = distance
                corresponding_xaxis = xaxis
        return corresponding_xaxis

    def find_closest_label(self, yaxis_mid_x, labels):
        min_distance = float('inf')
        corresponding_label = None
        for label in labels:
            label_mid_x = (label[0] + label[2]) / 2
            distance = abs(yaxis_mid_x - label_mid_x)
            if distance < min_distance:
                min_distance = distance
                corresponding_label = label
        return corresponding_label

    def find_group_bars(self, yaxis, xaxis, bars):
        group_bars = []
        for bar in bars:
            bar_mid_x = (bar[0] + bar[2]) / 2
            bar_mid_y = (bar[1] + bar[3]) / 2
            if yaxis[1] <= bar_mid_y <= yaxis[3] and xaxis[0] <= bar_mid_x <= xaxis[2]:
                group_bars.append(bar)
        return group_bars

    def extract_text_from_image(self, image, bbox, rotate=False):
        x1, y1, x2, y2 = map(int, bbox[:4])
        cropped_image = image[y1:y2, x1:x2]
        if rotate:
            cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)
        text = pytesseract.image_to_string(cropped_image, config='--psm 6')
        return text.strip()