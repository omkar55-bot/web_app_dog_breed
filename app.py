from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# Create the Flask app
app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained MobileNetV2 model
MODEL_PATH = 'model/20250115-10391736955548-10000-NasNetLarge_Adam_8050_val.h5'
model = load_model(MODEL_PATH)

# Dog breed class names (update as per your dataset)
CLASS_NAMES = ['Labrador', 'German Shepherd', 'Bulldog', 'Golden Retriever', 'Beagle', 'Poodle', 'Other']

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected!", 400

    # Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Load and preprocess the image
    img = Image.open(filepath).resize((224, 224))  # Resize to match MobileNetV2 input size
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = preprocess_input(img_array)  # Normalize image

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]

    return render_template('index.html', prediction=predicted_class, image_url=file.filename)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create uploads folder if not exists
    app.run(debug=True)
