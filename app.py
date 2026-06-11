import os
import cv2
import numpy as np
import easyocr
import joblib
import json
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Import our advanced feature module
from advanced_features import (
    remove_lines_from_page,
    segment_word_into_letters,
    extract_stroke_features,
    CNNEmbeddingExtractor,
    analyze_spelling,
    analyze_layout,
    generate_and_save_gradcam
)

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'crops'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'gradcam'), exist_ok=True)

# Load Keras CNN model from model folder
MODEL_PATH = os.path.join("model", "best_dyslexia_handwriting_model.keras")
model = load_model(MODEL_PATH)

# Load Meta-Classifier (SVM / Random Forest) if available
META_CLASSIFIER_PATH = os.path.join("model", "meta_classifier.pkl")
meta_classifier = None
if os.path.exists(META_CLASSIFIER_PATH):
    try:
        meta_classifier = joblib.load(META_CLASSIFIER_PATH)
        print("Meta-Classifier loaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to load Meta-Classifier: {e}")

# Instantiate the CNN feature embedding extractor
embedding_extractor = CNNEmbeddingExtractor(model)

# EasyOCR reader
reader = easyocr.Reader(['en'])

def preprocess_image(image_binary):
    """
    Prepares a binary character image for model input: 
    Resize to 28x28, duplicate channels to BGR (3 channels), and normalize.
    """
    image_resized = cv2.resize(image_binary, (28, 28))
    image_bgr = cv2.merge([image_resized, image_resized, image_resized])
    image_normalized = image_bgr / 255.0
    return np.expand_dims(image_normalized, axis=0)  # Shape: (1, 28, 28, 3)

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(x) for x in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())
    else:
        return obj

def run_diagnostic_pipeline(file_path, filename, questionnaire):
    image_raw = cv2.imread(file_path)
    if image_raw is None:
        return None
        
    # Erase horizontal ruling lines from the full page first
    image = remove_lines_from_page(image_raw)
    cv2.imwrite(file_path, image) # Overwrite on disk with cleaned version
    
    results = reader.readtext(image, detail=1, paragraph=False)
    
    letter_reports = []
    normal_chars = []
    corrected_chars = []
    reversal_chars = []
    
    word_centroids = []
    letter_bboxes = []
    slant_angles = []
    transcribed_words = []
    
    letter_counter = 0

    for idx, (bbox, text, prob) in enumerate(results):
        transcribed_words.append(text)
        
        x_center = (bbox[0][0] + bbox[2][0]) // 2
        y_center = (bbox[0][1] + bbox[2][1]) // 2
        word_centroids.append((x_center, y_center))
        
        x_min = int(max(0, min(bbox[0][0], bbox[3][0])))
        x_max = int(min(image.shape[1], max(bbox[1][0], bbox[2][0])))
        y_min = int(max(0, min(bbox[0][1], bbox[1][1])))
        y_max = int(min(image.shape[0], max(bbox[2][1], bbox[3][1])))
        
        word_crop = image[y_min:y_max, x_min:x_max]
        if word_crop.size == 0:
            continue
        
        segmented_letters = segment_word_into_letters(word_crop)
        
        for letter_idx, (letter_crop, local_bbox) in enumerate(segmented_letters):
            lx, ly, lw, lh = local_bbox
            global_x = x_min + lx
            global_y = y_min + ly
            letter_bboxes.append((global_x, global_y, lw, lh))
            
            char = text[letter_idx] if letter_idx < len(text) else "?"
            
            if char.isupper() or char.islower():
                letter_gray = cv2.cvtColor(letter_crop, cv2.COLOR_BGR2GRAY)
                _, letter_binary = cv2.threshold(letter_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                feat_dict = extract_stroke_features(letter_binary)
                feat_vector = np.array([
                    feat_dict["Aspect Ratio"],
                    feat_dict["Black-White Ratio"],
                    feat_dict["Stroke Thickness"],
                    feat_dict["Slant Angle"]
                ])
                slant_angles.append(feat_dict["Slant Angle"])
                
                processed_img = preprocess_image(letter_binary)
                
                if meta_classifier is not None:
                    cnn_emb = embedding_extractor.extract(processed_img)
                    fused_features = np.concatenate([cnn_emb, feat_vector]).reshape(1, -1)
                    
                    if hasattr(meta_classifier, "predict_proba"):
                        probs = meta_classifier.predict_proba(fused_features)[0]
                    else:
                        pred = meta_classifier.predict(fused_features)[0]
                        probs = np.zeros(3)
                        probs[pred] = 1.0
                else:
                    probs = model.predict(processed_img)[0]
                    
                class_idx = np.argmax(probs)
                confidence = float(probs[class_idx])
                
                if class_idx in (1, 2) and confidence < 0.82:
                    class_idx = 0
                    confidence = float(probs[0])
                    
                crop_filename = f"letter_{letter_counter}_{char}.png"
                crop_path = os.path.join(app.config['UPLOAD_FOLDER'], 'crops', crop_filename)
                cv2.imwrite(crop_path, letter_gray)
                
                gradcam_filename = None
                if class_idx in (1, 2):
                    gradcam_filename = f"gradcam_{letter_counter}_{char}.png"
                    gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], 'gradcam', gradcam_filename)
                    generate_and_save_gradcam(letter_gray, model, gradcam_path, pred_index=class_idx)
                    
                report = {
                    "char": char,
                    "class": ["Normal", "Corrected", "Reversal"][class_idx],
                    "confidence": confidence,
                    "crop_path": f"uploads/crops/{crop_filename}",
                    "gradcam_path": f"uploads/gradcam/{gradcam_filename}" if gradcam_filename else None,
                    "features": feat_dict
                }
                letter_reports.append(report)
                
                if class_idx == 0:
                    normal_chars.append(char)
                elif class_idx == 1:
                    corrected_chars.append(char)
                elif class_idx == 2:
                    reversal_chars.append(char)
                    
                letter_counter += 1

    spelling_report = analyze_spelling(transcribed_words)
    layout_report = analyze_layout(word_centroids, letter_bboxes, slant_angles)
    
    total_analyzed = len(normal_chars) + len(corrected_chars) + len(reversal_chars)
    dyslexia_percentage = 0.0
    if total_analyzed > 0:
        dyslexic_traits = len(reversal_chars) * 1.0 + len(corrected_chars) * 0.5
        base_percentage = (dyslexic_traits / total_analyzed) * 100.0
        
        if layout_report["Baseline Deviation"] > 15.0:
            base_percentage += 15.0
            
        words_checked = spelling_report.get('WordsChecked', 0)
        spelling_issues = len(spelling_report.get('Diagnostics', []))
        if words_checked > 0:
            misspelling_rate = spelling_issues / words_checked
            base_percentage += misspelling_rate * 30.0
        
        dyslexia_percentage = min(100.0, base_percentage)
    
    dyslexic_detected = dyslexia_percentage >= 20.0
    
    return {
        "filename": filename,
        "normal": normal_chars,
        "corrected": corrected_chars,
        "reversal": reversal_chars,
        "dyslexic_detected": dyslexic_detected,
        "dyslexia_percentage": dyslexia_percentage,
        "letter_reports": letter_reports,
        "spelling_report": spelling_report,
        "layout_report": layout_report,
        "questionnaire": questionnaire
    }

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

            age = request.form.get('age', 'N/A')
            gender = request.form.get('gender', 'N/A')
            reading_speed = request.form.get('reading_speed', 'Normal')
            letter_reversals_freq = request.form.get('letter_reversals_freq', 'Never')
            left_right_difficulty = request.form.get('left_right_difficulty', 'No')
            spelling_difficulty = request.form.get('spelling_difficulty', 'Low')
            
            questionnaire = {
                "age": age,
                "gender": gender,
                "reading_speed": reading_speed,
                "letter_reversals_freq": letter_reversals_freq,
                "left_right_difficulty": left_right_difficulty,
                "spelling_difficulty": spelling_difficulty
            }

            data = run_diagnostic_pipeline(file_path, filename, questionnaire)
            if data is None:
                return redirect(request.url)
            
            # Save results to JSON file
            json_path = file_path + '.json'
            try:
                serializable_data = make_json_serializable(data)
                with open(json_path, 'w') as f:
                    json.dump(serializable_data, f, indent=4)
            except Exception as e:
                print(f"Warning: Failed to save JSON report: {e}")

            return render_template('results.html', **data)
            
    # Scan uploads directory for previous files to show
    previous_uploads = []
    uploads_dir = app.config['UPLOAD_FOLDER']
    if os.path.exists(uploads_dir):
        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
        for f in os.listdir(uploads_dir):
            f_path = os.path.join(uploads_dir, f)
            if os.path.isfile(f_path) and f.lower().endswith(valid_extensions):
                name_clean = f.split('.')[0].replace('_', ' ').replace('-', ' ').title()
                
                # Check for corresponding JSON file
                json_path = f_path + '.json'
                status = "pending"
                risk_pct = None
                age_info = None
                
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as jf:
                            jsdata = json.load(jf)
                            status = "analyzed"
                            risk_pct = jsdata.get("dyslexia_percentage", 0.0)
                            age_info = jsdata.get("questionnaire", {}).get("age", "N/A")
                    except Exception as e:
                        print(f"Error reading JSON for {f}: {e}")
                
                previous_uploads.append({
                    "filename": f,
                    "name": name_clean,
                    "url": f"/static/uploads/{f}",
                    "status": status,
                    "risk_percentage": risk_pct,
                    "age": age_info
                })
                
    return render_template('index.html', previous_uploads=previous_uploads)

@app.route('/analyze/<filename>')
def analyze_existing(filename):
    filename = secure_filename(filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return redirect(url_for('index'))
        
    # Check if JSON report already exists, redirect to report view
    json_path = file_path + '.json'
    if os.path.exists(json_path):
        return redirect(url_for('view_report', filename=filename))
        
    # Pre-populate dummy questionnaire for existing uploads
    questionnaire = {
        "age": "10",
        "gender": "Female",
        "reading_speed": "Yes",
        "letter_reversals_freq": "Often",
        "left_right_difficulty": "Yes",
        "spelling_difficulty": "High"
    }
    
    data = run_diagnostic_pipeline(file_path, filename, questionnaire)
    if data is None:
        return redirect(url_for('index'))
        
    # Save results to JSON file
    try:
        serializable_data = make_json_serializable(data)
        with open(json_path, 'w') as f:
            json.dump(serializable_data, f, indent=4)
    except Exception as e:
        print(f"Warning: Failed to save JSON report: {e}")
        
    return render_template('results.html', **data)

@app.route('/report/<filename>')
def view_report(filename):
    filename = secure_filename(filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    json_path = file_path + '.json'
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return render_template('results.html', **data)
        except Exception as e:
            print(f"Error reading JSON report: {e}")
            
    # Fallback to run prediction if report not found
    return redirect(url_for('analyze_existing', filename=filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
