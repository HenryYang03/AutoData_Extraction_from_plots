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
