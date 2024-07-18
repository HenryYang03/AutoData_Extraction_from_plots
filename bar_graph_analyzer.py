import cv2
import torch
import pytesseract
import imutils
import re

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
        uptails = [d for d in detections if self.class_names[int(d[5])] == 'uptail']
        return bars, yaxes, xaxes, origins, ymaxes, labels, uptails

    def group_bars(self, detections, image):
        bars, yaxes, xaxes, origins, ymaxes, labels, uptails = self.filter_detections(detections)

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
                label_text = self.extract_text(image, corresponding_label, rotate=True)
                group_bars = self.find_group_bars(yaxis, corresponding_xaxis, bars)
                group_uptails = self.find_group_bars(yaxis, corresponding_xaxis, uptails)
                bar_groups[label_text] = (group_bars, group_uptails, yaxis, corresponding_xaxis, corresponding_origin, corresponding_ymax)

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

    def preprocess_image(self, image, bbox, for_numbers = False, padding = 4, save_images = False):
        x1, y1, x2, y2 = map(int, bbox[:4])
        height, width = image.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        cropped_image = image[y1:y2, x1:x2]

        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        resized_image = imutils.resize(gray_image, width=500)

        _, binary_image = cv2.threshold(resized_image, 175, 255, cv2.THRESH_BINARY_INV)

        if for_numbers:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            morph_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
            inverted_image = 255 - morph_image
            blurred_image = cv2.GaussianBlur(inverted_image, (5, 5), 0)

            if save_images:
                cv2.imwrite('blurred_image.png', blurred_image)
            return blurred_image

        return binary_image

    def extract_text(self, image, bbox, rotate = False):
        preprocessed_image = self.preprocess_image(image, bbox)
        if rotate:
            preprocessed_image = cv2.rotate(preprocessed_image, cv2.ROTATE_90_CLOCKWISE)

        config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(preprocessed_image, config = config)
        return text.strip()

    def extract_numbers(self, image, bbox, rotate = False):
        preprocessed_image = self.preprocess_image(image, bbox, for_numbers = True, save_images = True)
        if rotate:
            preprocessed_image = cv2.rotate(preprocessed_image, cv2.ROTATE_90_CLOCKWISE)

        custom_config = r'--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789.'
        text = pytesseract.image_to_string(preprocessed_image, config = custom_config)
        print(text)

        numbers = re.findall(r'\d+\.\d+|\d+', text)
        return numbers[0] if numbers else ''

    def calculate_heights(self, bar_groups, image):
        results = {}
        for label, (group_bars, group_uptails, yaxis, xaxis, origin, ymax) in bar_groups.items():
            sorted_bars = sorted(group_bars, key=lambda bar: bar[0])
            sorted_uptails = sorted(group_uptails, key=lambda uptail: uptail[0])

            origin_value = self.extract_numbers(image, origin)
            ymax_value = self.extract_numbers(image, ymax)

            try:
                origin_value = float(origin_value)
                ymax_value = float(ymax_value)
            except ValueError:
                raise ValueError("Can't convert the orgin and ymax to number")

            yaxis_height = yaxis[1] - yaxis[3]
            scale_factor = (ymax_value - origin_value) / yaxis_height

            bar_heights = []
            for bar in sorted_bars:
                bar_ymax = bar[1]# y-coordinate of the top of the bar
                bar_ymin = bar[3]
                #yaxis[3]
                height = (bar_ymax - bar_ymin) * scale_factor + origin_value  # Calculate the true height
                bar_heights.append(height)

            uptail_heights = []
            for uptail in sorted_uptails:
                uptail_ymax = uptail[1]
                uptail_ymin = uptail[3]
                height = (uptail_ymax - uptail_ymin) * scale_factor
                uptail_heights.append(height)

            results[label] = {
                "bar_heights": bar_heights,
                "uptail_heights": uptail_heights,
                "origin_value": origin_value,
                "ymax_value": ymax_value,
            }

        return results

if __name__ == "__main__":
    model_path = 'models/best.pt'
    class_names = ['label', 'ymax', 'origin', 'yaxis', 'bar', 'uptail', 'legend', 'legend_group', 'xaxis', 'x_group']
    analyzer = BarGraphAnalyzer(model_path, class_names, pytesseract_cmd='/opt/homebrew/bin/tesseract')

    image_path = '/Users/mohanyang/Desktop/new.png'
    bar_graph_heights = analyzer.analyze_image(image_path)
    print("Relative heights of the bars for each bar graph:", bar_graph_heights)