import os
import cv2
import easyocr
import numpy as np
import joblib
from tensorflow.keras.models import load_model

from advanced_features import (
    segment_word_into_letters,
    extract_stroke_features,
    CNNEmbeddingExtractor
)

def preprocess_image(image_binary):
    image_resized = cv2.resize(image_binary, (28, 28))
    image_bgr = cv2.merge([image_resized, image_resized, image_resized])
    image_normalized = image_bgr / 255.0
    return np.expand_dims(image_normalized, axis=0)

print("Loading Models...")
model = load_model("model/best_dyslexia_handwriting_model.keras")
meta_classifier = joblib.load("model/meta_classifier.pkl")
embedding_extractor = CNNEmbeddingExtractor(model)
reader = easyocr.Reader(['en'])

image_path = "testing_image/WhatsApp Image 2024-11-06 at 20.23.19.jpeg"
print(f"Testing on image: {image_path}")

image = cv2.imread(image_path)
if image is None:
    print("Could not load image.")
    exit(1)

results = reader.readtext(image, detail=1, paragraph=False)

print("\n--- Detection Results ---")
for idx, (bbox, text, prob) in enumerate(results):
    x_min = int(max(0, min(bbox[0][0], bbox[3][0])))
    x_max = int(min(image.shape[1], max(bbox[1][0], bbox[2][0])))
    y_min = int(max(0, min(bbox[0][1], bbox[1][1])))
    y_max = int(min(image.shape[0], max(bbox[2][1], bbox[3][1])))
    word_crop = image[y_min:y_max, x_min:x_max]
    if word_crop.size == 0: continue
    
    segmented_letters = segment_word_into_letters(word_crop)
    print(f"\nWord detected: '{text}' (Confidence: {prob:.2f})")
    
    for letter_idx, (letter_crop, local_bbox) in enumerate(segmented_letters):
        char = text[letter_idx] if letter_idx < len(text) else "?"
        if char.isupper() or char.islower():
            letter_gray = cv2.cvtColor(letter_crop, cv2.COLOR_BGR2GRAY)
            _, letter_binary = cv2.threshold(letter_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            feat_dict = extract_stroke_features(letter_binary)
            feat_vector = np.array([feat_dict["Aspect Ratio"], feat_dict["Black-White Ratio"], feat_dict["Stroke Thickness"], feat_dict["Slant Angle"]])
            processed_img = preprocess_image(letter_binary)
            
            cnn_emb = embedding_extractor.extract(processed_img)
            fused_features = np.concatenate([cnn_emb, feat_vector]).reshape(1, -1)
            
            probs = meta_classifier.predict_proba(fused_features)[0]
            class_idx = np.argmax(probs)
            confidence = float(probs[class_idx])
            
            if class_idx in (1, 2) and confidence < 0.82:
                class_idx = 0
                confidence = float(probs[0])
                
            pred_class = ["Normal", "Corrected", "Reversal"][class_idx]
            
            print(f"  Char '{char}': {pred_class} (Conf: {confidence*100:.1f}%) - Features: [AR={feat_dict['Aspect Ratio']:.2f}, Slant={feat_dict['Slant Angle']:.1f}, Thick={feat_dict['Stroke Thickness']:.2f}]")
