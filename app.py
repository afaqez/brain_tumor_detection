from flask import Flask, render_template, request, jsonify, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
import firebase_admin
from flask_cors import CORS
from firebase_admin import credentials, auth
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

app = Flask(__name__, template_folder='template')
CORS(app)  

app = Flask(__name__)
CORS(app, resources={r"/upload": {"origins": "*"}})  

# Initialize Firebase Admin SDK
cred = credentials.Certificate('brain-tumour-detection-7255a-firebase-adminsdk-wcgc3-6184a02a1d.json')
firebase_admin.initialize_app(cred)

# Load the trained models
vgg_model = load_model('models/vgg.h5')
inception_model = load_model('models/inceptionv3_brain_tumor.h5')
cnn_model = load_model('models/cnn_brain_tumor.h5')
resnet_model = load_model('models/resnet_brain_tumor.h5')

# Preprocess image functions
def preprocess_image_vgg(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((32, 32))
    image = np.array(image) / 255.0
    return image

def preprocess_image_inception_v3(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((75, 75))
    image = np.array(image) / 255.0
    return image

def preprocess_image_cnn(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((32, 32))
    image = np.array(image) / 255.0
    return image

def preprocess_image_resnet(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return image

# Verify Firebase ID token
def verify_id_token(id_token):
    try:
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except:
        return None

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/contact')
def view_contact():
    return render_template('contact.html')

@app.route('/detection')
def view_detection():
    return render_template('detection.html')

@app.route('/first')
def view_first():
    return render_template('first.html')

@app.route('/gallery')
def view_gallery():
    return render_template('gallery.html')

@app.route('/login')
def view_login():
    return render_template('login.html')

@app.route('/performance-analysis')
def view_performance():
    return render_template('performance-analysis.html')

@app.route('/result')
def view_result():
    return render_template('result.html')

@app.route('/settings')
def view_settings():
    return render_template('settings.html')

@app.route('/signup')
def view_signup():
    return render_template('signup.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    id_token = request.headers.get('Authorization')
    # if not id_token:
    #     return jsonify({'error': 'Authorization token not provided'}), 403

    # user = verify_id_token(id_token)
    # if not user:
    #     return jsonify({'error': 'Invalid or expired token'}), 403

    if 'mriImage' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    if 'modelSelect' not in request.form:
        return jsonify({'error': 'No model selected'}), 400

    mri_image = request.files['mriImage']
    selected_model = request.form['modelSelect']

    if selected_model not in ['vgg', 'inception', 'cnn', 'resnet']:
        return jsonify({'error': 'Invalid model selected'}), 400

    image_path = os.path.join('static/uploads', mri_image.filename)
    mri_image.save(image_path)

    with open(image_path, 'rb') as f:
        image_data = f.read()

    if selected_model == 'vgg':
        preprocessed_image = preprocess_image_vgg(image_data)
        prediction = vgg_model.predict(np.expand_dims(preprocessed_image, axis=0))
    elif selected_model == 'inception':
        preprocessed_image = preprocess_image_inception_v3(image_data)
        prediction = inception_model.predict(np.expand_dims(preprocessed_image, axis=0))
    elif selected_model == 'cnn':
        preprocessed_image = preprocess_image_cnn(image_data)
        prediction = cnn_model.predict(np.expand_dims(preprocessed_image, axis=0))
    elif selected_model == 'resnet':
        preprocessed_image = preprocess_image_resnet(image_data)
        prediction = resnet_model.predict(np.expand_dims(preprocessed_image, axis=0))

    predicted_class = np.argmax(prediction)
    prediction_data = {
        'filename': image_path,
        'selected_model': selected_model,
        'predicted_class': int(predicted_class)
    }

    with open('result.json', 'w') as file:
        file.write(json.dumps(prediction_data))

    return jsonify(prediction_data)

@app.route('/get_result')
def get_result():
    with open('result.json', 'r') as file:
        prediction_data = json.load(file)
    return jsonify(prediction_data)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
