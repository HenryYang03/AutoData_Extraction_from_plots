from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import cv2
from PIL import Image
import torch
from bar_graph_analyzer import BarGraphAnalyzer

# Initialize Flask application
app = Flask(__name__)
app.secret_key = "b'\xca\xa4\xf2\x80!\xfe\x85\xba\xd7\xcf\xe7\xc9\xf1)I\xac\x10Y5M\x95\xed\xfb\xc4'"
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


@app.route('/bar_analyzer', methods = ['GET', 'POST'])
def bar_analyzer():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                bar_graph_heights = analyzer.analyze_image(filepath)
                detections = analyzer.model(filepath).xyxy[0].numpy()
                detection_image_path = draw_detections(filepath, detections)
                return render_template('bar_analyzer.html', filename = filename,
                                       results = bar_graph_heights, detection_image_path = detection_image_path)
            except ValueError as e:
                flash(str(e))
                return redirect(request.url)

    return render_template('bar_analyzer.html')

@app.route('/box_analyzer')
def box_analyzer():
    return render_template('box_analyzer.html')

if __name__ == "__main__":
    app.run(debug=True)