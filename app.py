from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import easyocr
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# Load model from model folder
MODEL_PATH = os.path.join("model", "best_dyslexia_handwriting_model.keras")
model = load_model(MODEL_PATH)

# EasyOCR reader
reader = easyocr.Reader(['en'])

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Preprocess image for model
def preprocess_image(image):
    image_resized = cv2.resize(image, (28, 28))
    if len(image_resized.shape) == 2:
        image_resized = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2BGR)
    image_normalized = image_resized / 255.0
    return np.expand_dims(image_normalized, axis=0)  # Shape: (1, 28, 28, 3)

# Segment letters using EasyOCR
def segment_letters_with_easyocr(image_path):
    image = cv2.imread(image_path)
    results = reader.readtext(image, detail=1, paragraph=False)

    letter_images = []
    predictions = []

    for _, text, _ in results:
        for char in text:
            if char.isupper():
                processed_image = preprocess_image(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                prediction = model.predict(processed_image)
                predictions.append((char, prediction[0][0]))
                letter_images.append(char)

    return letter_images, predictions

# Categorize predictions
def categorize_predictions(predictions):
    normal, corrected, reversal = [], [], []
    for char, prob in predictions:
        if prob < 0.3:
            normal.append(char)
        elif 0.3 <= prob < 0.7:
            corrected.append(char)
        else:
            reversal.append(char)
    return normal, corrected, reversal

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            letter_images, predictions = segment_letters_with_easyocr(file_path)
            normal, corrected, reversal = categorize_predictions(predictions)

            dyslexic_detected = any(prob > 0.5 for _, prob in predictions)

            return render_template(
                'results.html',  # ✅ matches your file name
                filename=filename,
                normal=normal,
                corrected=corrected,
                reversal=reversal,
                dyslexic_detected=dyslexic_detected
            )
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return url_for('static', filename='uploads/' + filename)

if __name__ == "__main__":
    app.run(debug=True)
