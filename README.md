# AI-Powered Dyslexia Detection & Assessment System

A premium, full-stack healthcare-tech web application that leverages Deep Learning and Computer Vision to predict dyslexia risk by analyzing handwriting samples. The system evaluates morphological character anomalies, spatial layout variations, and spelling discrepancies, combining them with a behavioral demographic profile to generate a comprehensive diagnostic report.

## ✨ Key Features
- **Computer Vision Pipeline:** Automatically strips ruling lines from lined paper and segments handwritten text into individual characters using OpenCV and EasyOCR.
- **Hybrid ML Classification:** A custom TensorFlow/Keras CNN combined with a Scikit-Learn meta-classifier detects character reversals (e.g., b/d, p/q) with high confidence.
- **Explainable AI (XAI):** Generates Grad-CAM heatmaps to visually explain to users and clinicians why a specific character was flagged.
- **Spatial & Spelling Diagnostics:** Calculates baseline drift, spacing variance, and utilizes NLP heuristics to catch spelling transposition anomalies.
- **Premium Frontend:** A fully responsive, dark-themed glassmorphism interface built with Flask, HTML, CSS, and JS.
- **Interactive Archive Gallery:** Dynamically caches previous assessments in JSON format, allowing users to browse and re-analyze past uploads via a beautiful showcase grid.

## 🛠️ Technology Stack
- **Backend Framework:** Python, Flask
- **Deep Learning / ML:** TensorFlow/Keras, Scikit-Learn, Joblib
- **Computer Vision & OCR:** OpenCV, EasyOCR, NumPy
- **Frontend UI:** HTML5, Vanilla CSS3 (Glassmorphism), JavaScript, Jinja2

## 🚀 Setup Instructions

### Prerequisites
Make sure you have Python 3.9+ installed on your system.

### 1. Clone the repository
```bash
git clone https://github.com/ayaz1017/Dyslexia-Prediction-Using-DL.git
cd Dyslexia-Prediction-Using-DL
```

### 2. Install dependencies
Install the required Python packages using `pip`:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Start the Flask development server:
```bash
python app.py
```

### 4. Access the Web Interface
Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

## 📸 Usage & Workflow
1. **Interactive Form:** Complete the 3-step screening profile (demographics, behavioral checklist, and file upload).
2. **Upload Sample:** Upload an image of handwriting on lined or blank paper.
3. **Automated Analysis:** The system preprocesses the image, extracts neural embeddings, and computes the risk score.
4. **Diagnostic Dashboard:** View the final calculated Dyslexia Risk Percentage, the spatial layout metrics, spelling check details, and visual character heatmaps.