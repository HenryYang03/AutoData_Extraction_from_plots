from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
from PIL import Image
import torch
from bar_graph_analyzer import BarGraphAnalyzer

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load the YOLO model and class names
model_path = 'models/test_model.pt'
class_names = ['label', 'ymax', 'origin', 'yaxis', 'bar', 'uptail', 'legend', 'legend_group', 'xaxis', 'x_group']
analyzer = BarGraphAnalyzer(model_path, class_names, pytesseract_cmd='tesseract')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def save_file(file):
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filepath, filename

def analyze_image(filepath):
    return analyzer.analyze_image(filepath)

def draw_detections(image_path, detections):
    image = cv2.imread(image_path)
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        label = class_names[int(cls)]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_' + os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    return output_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filepath, filename = save_file(file)
        bar_graph_heights = analyze_image(filepath)
        detections = analyzer.model(filepath).xyxy[0].numpy()  # Extract detections
        detection_image_path = draw_detections(filepath, detections)
        return render_template('index.html', filename=filename, results=bar_graph_heights, detection_image_path=detection_image_path)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)