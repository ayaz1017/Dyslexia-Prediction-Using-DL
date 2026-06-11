import os
import sys
import glob
import subprocess
import joblib
import cv2
import numpy as np
import tensorflow as tf
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Import our helper functions
from advanced_features import extract_stroke_features, CNNEmbeddingExtractor

def extract_dataset():
    """
    Extracts the Gambo dataset from the password-protected RAR file using 7z.
    """
    rar_path = os.path.join("Ayaz-Line_Segmentation", "Dataset Dyslexia_Password WanAsy321 (1)(1).rar")
    if not os.path.exists(rar_path):
        print(f"Error: RAR file not found at {rar_path}")
        sys.exit(1)
        
    print("Extracting Gambo dataset using 7-Zip...")
    # We use the absolute path to 7z we found earlier
    sz_path = r"C:\Program Files\7-Zip\7z.exe"
    
    # Extract only the Gambo folder
    cmd = [
        sz_path, 
        "x", 
        rar_path, 
        "-pWanAsy321", 
        "-oextracted_gambo", 
        "-y",
        "Gambo/*"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("Extraction completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during extraction: {e}")
        sys.exit(1)

def build_dataset(gambo_dir, model, sample_limit=2000):
    """
    Loads images, extracts fused features (CNN embedding + handcrafted features),
    and builds X (features) and y (labels) arrays.
    
    Args:
        gambo_dir: Path to the extracted 'Gambo' directory.
        model: Loaded Keras model.
        sample_limit: Max samples per class to prevent long CPU runs.
    """
    classes = {"Normal": 0, "Corrected": 1, "Reversal": 2}
    extractor = CNNEmbeddingExtractor(model)
    
    X = []
    y = []
    
    # We check Train directory
    train_dir = os.path.join(gambo_dir, "Train")
    if not os.path.exists(train_dir):
        # Check if Gambo is nested directly under extracted_gambo
        train_dir = os.path.join(gambo_dir, "Gambo", "Train")
        if not os.path.exists(train_dir):
            print(f"Error: Train directory not found in {gambo_dir}")
            return None, None
            
    print(f"Extracting features (limit: {sample_limit} per class)...")
    for class_name, label in classes.items():
        class_path = os.path.join(train_dir, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Class directory {class_path} does not exist.")
            continue
            
        # Get list of images
        img_paths = glob.glob(os.path.join(class_path, "*.*"))
        # Filter for valid image extensions
        img_paths = [p for p in img_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        # Shuffle and subset
        np.random.shuffle(img_paths)
        img_paths = img_paths[:sample_limit]
        
        print(f"Processing class '{class_name}' ({len(img_paths)} images)...")
        
        for idx, img_path in enumerate(img_paths):
            try:
                # Load image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                    
                # Resize and binarize using Otsu's thresholding
                resized = cv2.resize(img, (28, 28))
                _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                
                # 1. Extract Handcrafted Stroke Features
                feat_dict = extract_stroke_features(binary)
                feat_vector = np.array([
                    feat_dict["Aspect Ratio"],
                    feat_dict["Black-White Ratio"],
                    feat_dict["Stroke Thickness"],
                    feat_dict["Slant Angle"]
                ])
                
                # 2. Extract CNN Penultimate Layer Embedding
                # Prepare image for model input shape (1, 28, 28, 3)
                bgr = cv2.merge([resized, resized, resized])
                normalized = bgr / 255.0
                img_input = np.expand_dims(normalized, axis=0)
                
                cnn_embedding = extractor.extract(img_input)
                
                # 3. Concatenate CNN + Handcrafted features
                fused_vector = np.concatenate([cnn_embedding, feat_vector])
                
                X.append(fused_vector)
                y.append(label)
                
                if (idx + 1) % 500 == 0:
                    print(f"  Processed {idx + 1}/{len(img_paths)}...")
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
                
    return np.array(X), np.array(y)

def main():
    # 1. Extract the dataset from the RAR file
    gambo_dir = "extracted_gambo"
    if not os.path.exists(gambo_dir) or not os.path.exists(os.path.join(gambo_dir, "Gambo")):
        extract_dataset()
    else:
        print("Dataset already extracted, skipping extraction.")
        
    # Find the Gambo folder path
    gambo_path = os.path.join(gambo_dir, "Gambo")
    if not os.path.exists(gambo_path):
        gambo_path = gambo_dir
        
    # 2. Load the Keras CNN model
    keras_model_path = os.path.join("model", "best_dyslexia_handwriting_model.keras")
    if not os.path.exists(keras_model_path):
        print(f"Error: Keras model not found at {keras_model_path}")
        sys.exit(1)
        
    print("Loading pre-trained Keras model...")
    model = tf.keras.models.load_model(keras_model_path)
    
    # 3. Extract features and build data matrices
    # We set a limit of 3000 samples per class to keep execution fast and prevent memory issues
    X, y = build_dataset(gambo_path, model, sample_limit=3000)
    if X is None or len(X) == 0:
        print("Error: No training features extracted.")
        sys.exit(1)
        
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # 4. Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Train the meta-classifier
    print("Training SVM Meta-Classifier...")
    # Random Forest is also a great alternative for non-linear decision boundaries
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # 6. Evaluate accuracy
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Corrected", "Reversal"]))
    
    # 7. Save model to disk
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    meta_path = os.path.join(model_dir, "meta_classifier.pkl")
    print(f"Saving meta-classifier to {meta_path}...")
    joblib.dump(clf, meta_path)
    print("Meta-Classifier training completed successfully!")

if __name__ == "__main__":
    main()
