from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import numpy as np
import requests
import os
import gdown


app = Flask(__name__)

# Define the path where the model will be saved after download
MODEL_PATH = "best_model.keras"
MODEL_URL = "https://drive.google.com/uc?id=1UeXqEXmkF3Zf4E8GhjnxPQA2enXgMkXD"

# Download the model from Google Drive if it's not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# Define the disease classes
DISEASE_CLASSES = [
    'Tomato___Late_blight', 'Tomato___healthy', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Potato___healthy', 'Corn_(maize)___Northern_Leaf_Blight',
    'Tomato___Early_blight', 'Tomato___Septoria_leaf_spot',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Strawberry___Leaf_scorch',
    'Peach___healthy', 'Apple___Apple_scab', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Bacterial_spot', 'Apple___Black_rot', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Peach___Bacterial_spot',
    'Apple___Cedar_apple_rust', 'Tomato___Target_Spot', 'Pepper,_bell___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Late_blight',
    'Tomato___Tomato_mosaic_virus', 'Strawberry___healthy', 'Apple___healthy',
    'Grape___Black_rot', 'Potato___Early_blight', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Common_rust_', 'Grape___Esca_(Black_Measles)', 'Raspberry___healthy',
    'Tomato___Leaf_Mold', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Pepper,_bell___Bacterial_spot', 'Corn_(maize)___healthy'
]

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Check if the file is included in the request
        if 'file' not in request.files or not request.files['file']:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        # 2. Validate the file type
        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

        # 3. Try opening the file as an image
        try:
            img = Image.open(file)
        except UnidentifiedImageError:
            return jsonify({'error': 'Invalid image file. Ensure the file is a valid image.'}), 400

        # Preprocess the image
        img = img.resize((224, 224))  # Match model's input size
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = float(np.max(prediction))

        # Map the predicted class index to the class name
        predicted_class_name = DISEASE_CLASSES[predicted_class_idx]

        # Return the result as JSON
        return jsonify({
            'predicted_class': predicted_class_name,
            'confidence': confidence
        })

    except Exception as e:
        # General error handler for unexpected issues
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

    """Download the model from Google Drive if it doesn't exist locally."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("Downloading model from Google Drive...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Model downloaded successfully to {MODEL_PATH}.")
        else:
            raise Exception(f"Failed to download the model. HTTP status code: {response.status_code}")
    else:
        print(f"Model already exists at {MODEL_PATH}.")

# Ensure the model is downloaded
download_model()

# Verify the file path
print(f"Checking if the model file exists at {MODEL_PATH}...")
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")

# Load the trained model
print("Loading the model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")


# Define the disease classes
DISEASE_CLASSES = [
    'Tomato___Late_blight', 'Tomato___healthy', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Potato___healthy', 'Corn_(maize)___Northern_Leaf_Blight',
    'Tomato___Early_blight', 'Tomato___Septoria_leaf_spot',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Strawberry___Leaf_scorch',
    'Peach___healthy', 'Apple___Apple_scab', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Bacterial_spot', 'Apple___Black_rot', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Peach___Bacterial_spot',
    'Apple___Cedar_apple_rust', 'Tomato___Target_Spot', 'Pepper,_bell___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Late_blight',
    'Tomato___Tomato_mosaic_virus', 'Strawberry___healthy', 'Apple___healthy',
    'Grape___Black_rot', 'Potato___Early_blight', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Common_rust_', 'Grape___Esca_(Black_Measles)', 'Raspberry___healthy',
    'Tomato___Leaf_Mold', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Pepper,_bell___Bacterial_spot', 'Corn_(maize)___healthy'
]

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Check if the file is included in the request
        if 'file' not in request.files or not request.files['file']:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        # 2. Validate the file type
        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

        # 3. Try opening the file as an image
        try:
            img = Image.open(file)
        except UnidentifiedImageError:
            return jsonify({'error': 'Invalid image file. Ensure the file is a valid image.'}), 400

        # Preprocess the image
        img = img.resize((224, 224))  # Match model's input size
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = float(np.max(prediction))

        # Map the predicted class index to the class name
        predicted_class_name = DISEASE_CLASSES[predicted_class_idx]

        # Return the result as JSON
        return jsonify({
            'predicted_class': predicted_class_name,
            'confidence': confidence
        })

    except Exception as e:
        # General error handler for unexpected issues
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
