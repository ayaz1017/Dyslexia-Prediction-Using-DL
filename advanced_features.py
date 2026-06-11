import os
import cv2
import numpy as np
import tensorflow as tf

# ==========================================
# STEP 2.0: FULL-PAGE LINE REMOVAL
# ==========================================

def remove_lines_from_page(image):
    """
    Detects and erases horizontal ruling lines from a full handwritten document page.
    Replaces the line pixels with white paper color.
    """
    if image is None or image.size == 0:
        return image
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Detect horizontal lines
    height, width = binary.shape
    kernel_width = max(100, int(width * 0.15))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    # Dilate vertically slightly to cover the thickness of the line (usually 3px)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    detected_lines = cv2.dilate(detected_lines, dilation_kernel, iterations=1)
    
    # Erase detected lines by painting them white in the original BGR image
    cleaned_image = image.copy()
    cleaned_image[detected_lines == 255] = [255, 255, 255]
    
    return cleaned_image


# ==========================================
# STEP 2.1: CONTOUR-BASED CHARACTER SEGMENTATION
# ==========================================

def segment_word_into_letters(word_image):
    """
    Segments a cropped word image into individual letters using contour detection.
    Also handles vertical projection profile split for overlapping/connected characters.
    Removes horizontal line noise from ruled paper.
    
    Args:
        word_image: cv2 BGR image containing a single word.
    Returns:
        List of tuples: (letter_crop_rgb, bbox_in_word) where bbox_in_word is (x, y, w, h)
    """
    if word_image is None or word_image.size == 0:
        return []
    
    # Make a copy to avoid modifying the source image directly
    word_image_clean = word_image.copy()

    gray = cv2.cvtColor(word_image_clean, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # --- Ruled Line Removal ---
    height, width = binary.shape
    # Detect horizontal lines that span a significant portion of the word width
    kernel_width = max(20, int(width * 0.35))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    # Dilate vertically to cover line thickness
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    detected_lines = cv2.dilate(detected_lines, dilation_kernel, iterations=1)
    
    # Clean the binary mask and paint lines white in BGR image
    binary_cleaned = cv2.subtract(binary, detected_lines)
    word_image_clean[detected_lines == 255] = [255, 255, 255]
    
    # 2. Find external contours representing strokes on the cleaned mask
    contours, _ = cv2.findContours(binary_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. Filter noise and extract initial bounding boxes
    bboxes = []
    min_area = 20  # Area threshold to filter out tiny dust/noise
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > min_area and w > 2 and h > 2:
            # Avoid horizontal line residues
            if w / h > 3.5 and h < 6:
                continue
            # Filter out tiny noise at crop boundaries
            if h < height * 0.12:
                continue
            bboxes.append((x, y, w, h))
            
    # 4. Sort bounding boxes from left-to-right
    bboxes = sorted(bboxes, key=lambda b: b[0])
    
    final_bboxes = []
    # 5. Handle connected characters via Vertical Projection Profile
    for (x, y, w, h) in bboxes:
        aspect = w / h if h > 0 else 0
        if aspect > 1.6:  # Threshold for connected character block
            binary_patch = binary_cleaned[y:y+h, x:x+w]
            proj = np.sum(binary_patch == 255, axis=0)
            
            splits = int(np.round(aspect))
            split_w = w // splits
            for s in range(splits):
                final_bboxes.append((x + s * split_w, y, split_w, h))
        else:
            final_bboxes.append((x, y, w, h))
            
    # 6. Crop letters from the cleaned BGR image
    letter_crops = []
    for (x, y, w, h) in final_bboxes:
        # Avoid boundary overflow
        x_min, y_min = max(0, x), max(0, y)
        x_max = min(word_image_clean.shape[1], x + w)
        y_max = min(word_image_clean.shape[0], y + h)
        
        letter_crop = word_image_clean[y_min:y_max, x_min:x_max]
        if letter_crop.size > 0:
            letter_crops.append((letter_crop, (x_min, y_min, w, h)))
            
    return letter_crops


# ==========================================
# STEP 2.2: HANDCRAFTED FEATURE EXTRACTION
# ==========================================

def extract_stroke_features(image_binary):
    """
    Extracts handcrafted structural features of a single character stroke.
    
    Args:
        image_binary: Grayscale inverted binary image (28x28) where stroke is 255.
    Returns:
        Dict of features: Aspect Ratio, Black-White Ratio, Stroke Thickness, Slant Angle
    """
    if len(image_binary.shape) == 3:
        image_binary = cv2.cvtColor(image_binary, cv2.COLOR_BGR2GRAY)
        
    height, width = image_binary.shape
    aspect_ratio = width / height if height > 0 else 0
    
    # Black vs White Pixels (stroke is 255, background is 0)
    white_pixels = np.sum(image_binary == 255)
    total_pixels = height * width
    black_pixels = total_pixels - white_pixels
    black_white_ratio = black_pixels / white_pixels if white_pixels > 0 else black_pixels
    
    # Stroke Thickness (Mean horizontal projection of stroke pixels)
    horizontal_projection = np.sum(image_binary, axis=1) / 255
    active_rows = horizontal_projection[horizontal_projection > 0]
    stroke_thickness = np.mean(active_rows) if len(active_rows) > 0 else 0
    
    # Slant Angle (Using Hough Lines on Canny Edges)
    edges = cv2.Canny(image_binary, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)
    slant_angle = 90.0  # Default vertical slant angle
    
    if lines is not None:
        angles = []
        for line in lines:
            theta = line[0][1]
            angle = np.degrees(theta)
            # Map theta close to 90 degrees deviation
            angles.append(angle)
        slant_angle = np.mean(angles)
        
    return {
        "Aspect Ratio": float(aspect_ratio),
        "Black-White Ratio": float(black_white_ratio),
        "Stroke Thickness": float(stroke_thickness),
        "Slant Angle": float(slant_angle)
    }




class CNNEmbeddingExtractor:
    """
    Slices the pre-trained Keras model at the penultimate layer ('dense_2') 
    to extract 128-dimensional deep learning feature embeddings.
    """
    def __init__(self, main_model):
        self.main_model = main_model
        try:
            # Slicing at the penultimate Dense layer 'dense_2'
            layer = main_model.get_layer('dense_2')
            self.embedding_model = tf.keras.models.Model(
                inputs=main_model.inputs, 
                outputs=layer.output
            )
        except Exception:
            # Fallback in case layer names differ
            dense_layers = [l for l in main_model.layers if 'dense' in l.name]
            self.embedding_model = tf.keras.models.Model(
                inputs=main_model.inputs, 
                outputs=dense_layers[-2].output
            )

    def extract(self, processed_image):
        """
        Extracts the 128D embedding vector.
        Args:
            processed_image: Shape (1, 28, 28, 3)
        Returns:
            Numpy array of shape (128,)
        """
        embeddings = self.embedding_model.predict(processed_image, verbose=0)
        return embeddings[0]


# ==========================================
# STEP 2.4: SPATIAL LAYOUT ANALYSIS
# ==========================================

def analyze_layout(word_centroids, letter_bboxes, slant_angles):
    """
    Analyzes global page metrics to identify dysgraphia/dyslexia spatial irregularities.
    Groups word centroids into text lines to accurately compute baseline deviation on multi-line documents.
    
    Args:
        word_centroids: List of (x, y) coordinates representing word centers.
        letter_bboxes: List of bounding boxes (x, y, w, h) of all letters.
        slant_angles: List of float slant angles.
    Returns:
        Dict: Baseline Deviation, Spacing Variance, Slant Variability
    """
    # 1. Baseline Deviation (Residual variance of fit line grouped by lines)
    baseline_deviation = 0.0
    if len(word_centroids) >= 3:
        # Group centroids into lines based on vertical proximity
        avg_h = np.mean([b[3] for b in letter_bboxes]) if len(letter_bboxes) > 0 else 30.0
        threshold_y = avg_h * 0.8  # Threshold to separate text lines
        
        # Sort centroids by y coordinate first to facilitate line grouping
        sorted_centroids = sorted(word_centroids, key=lambda c: c[1])
        
        lines = []
        if len(sorted_centroids) > 0:
            current_line = [sorted_centroids[0]]
            for c in sorted_centroids[1:]:
                # If the y-distance to the current line's average y is small, group it
                if abs(c[1] - np.mean([x[1] for x in current_line])) < threshold_y:
                    current_line.append(c)
                else:
                    lines.append(current_line)
                    current_line = [c]
            lines.append(current_line)
            
        # Fit a line for each text line with >= 2 words, and average their residuals
        line_stds = []
        for line in lines:
            if len(line) >= 2:
                # Sort horizontally left-to-right
                line = sorted(line, key=lambda c: c[0])
                x_coords = np.array([c[0] for c in line])
                y_coords = np.array([c[1] for c in line])
                
                # Need at least 2 points to fit a line
                if len(line) >= 2:
                    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
                    try:
                        m, c_val = np.linalg.lstsq(A, y_coords, rcond=None)[0]
                        y_pred = m * x_coords + c_val
                        residuals = y_coords - y_pred
                        line_stds.append(np.std(residuals))
                    except Exception:
                        pass
        if len(line_stds) > 0:
            baseline_deviation = float(np.mean(line_stds))
        else:
            baseline_deviation = 0.0
            
    # 2. Spacing Variance (Standard deviation of letter spacing gaps)
    spacing_variance = 0.0
    if len(letter_bboxes) >= 2:
        # Sort boxes by left-to-right x coordinate
        sorted_boxes = sorted(letter_bboxes, key=lambda b: b[0])
        spacings = []
        for i in range(len(sorted_boxes) - 1):
            gap = sorted_boxes[i+1][0] - (sorted_boxes[i][0] + sorted_boxes[i][2])
            # Filter out negative gaps due to overlapping lines
            spacings.append(max(0, gap))
        if len(spacings) > 0:
            spacing_variance = float(np.std(spacings))
            
    # 3. Slant Variability (Variance of slant angles across all characters)
    slant_variability = 0.0
    if len(slant_angles) >= 2:
        slant_variability = float(np.var(slant_angles))
        
    return {
        "Baseline Deviation": baseline_deviation,
        "Spacing Variance": spacing_variance,
        "Slant Variability": slant_variability
    }


# ==========================================
# STEP 2.5: NLP SPELLING DIAGNOSTICS
# ==========================================

# A small built-in list of common English words for spellcheck fallback
COMMON_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", 
    "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", 
    "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", 
    "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me", "when", 
    "make", "can", "like", "time", "no", "just", "him", "know", "take", "people", "into", 
    "year", "your", "good", "some", "could", "them", "see", "other", "than", "then", "now", 
    "look", "only", "come", "its", "over", "think", "also", "back", "after", "use", "two", 
    "how", "our", "work", "first", "well", "way", "even", "new", "want", "because", "any", 
    "these", "give", "day", "most", "us", "love", "school", "friend", "knows", "write", "chest",
    "sleep", "hand", "eye", "close", "simply", "where", "climate", "problem", "going", "without",
    "knowing", "problems", "pride", "loving", "intimate", "upon", "asleep", "eyes", "fall", "know"
}

def levenshtein_distance(s1, s2):
    """Calculates Levenshtein edit distance between two strings."""
    s1, s2 = s1.lower(), s2.lower()
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def is_valid_spelling(word, dictionary):
    """Checks spelling using common suffixes/plural stemming rules."""
    w = word.lower()
    if w in dictionary:
        return True
    # Plurals/Suffixes
    if w.endswith('s') and w[:-1] in dictionary:
        return True
    if w.endswith('es') and w[:-2] in dictionary:
        return True
    if w.endswith('ed') and w[:-2] in dictionary:
        return True
    if w.endswith('ed') and w[:-1] in dictionary:
        return True
    if w.endswith('ing'):
        if w[:-3] in dictionary:
            return True
        if w[:-3] + 'e' in dictionary:
            return True
    if w.endswith('ly') and w[:-2] in dictionary:
        return True
    return False

def analyze_spelling(transcribed_words):
    """
    Logs spelling metrics (Levenshtein distances) and checks for dyslexia error patterns:
    - Vowel omissions
    - Permutations (transpositions)
    """
    diagnostics = []
    total_dist = 0
    words_checked = 0
    
    for word in transcribed_words:
        word_clean = "".join([c for c in word if c.isalnum()]).lower()
        if not word_clean or len(word_clean) < 3:
            continue
            
        words_checked += 1
        if is_valid_spelling(word_clean, COMMON_WORDS):
            continue  # Correctly spelled
            
        # Find closest word in dictionary
        best_match = None
        min_dist = 999
        for dict_w in COMMON_WORDS:
            dist = levenshtein_distance(word_clean, dict_w)
            if dist < min_dist:
                min_dist = dist
                best_match = dict_w
                
        if min_dist <= 2 and best_match:  # Closest typo match
            total_dist += min_dist
            
            # Diagnose patterns
            is_transposition = sorted(word_clean) == sorted(best_match)
            
            # Check for vowel omission
            vowels = "aeiou"
            word_no_vowels = "".join([c for c in word_clean if c not in vowels])
            match_no_vowels = "".join([c for c in best_match if c not in vowels])
            is_vowel_omission = (word_no_vowels == match_no_vowels) and (len(word_clean) < len(best_match))
            
            error_type = "Typo"
            if is_transposition:
                error_type = "Transposition (letter swap)"
            elif is_vowel_omission:
                error_type = "Vowel Omission"
                
            diagnostics.append({
                "Transcribed": word,
                "Suggested": best_match,
                "EditDistance": min_dist,
                "Pattern": error_type
            })
            
    avg_edit_distance = total_dist / words_checked if words_checked > 0 else 0.0
    return {
        "Diagnostics": diagnostics,
        "AvgEditDistance": avg_edit_distance,
        "WordsChecked": words_checked
    }


# ==========================================
# STEP 2.6: EXPLAINABLE AI (GRAD-CAM)
# ==========================================

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_5", pred_index=None):
    """
    Generates a Grad-CAM heatmap showing structural focus areas on a specific character image.
    
    Args:
        img_array: Preprocessed image of shape (1, 28, 28, 3).
        model: The loaded Keras model.
        last_conv_layer_name: Name of the target convolutional layer.
        pred_index: Index of the class (0, 1, or 2) to visualize.
    Returns:
        2D heatmap array normalized between 0 and 1.
    """
    # Create a grad model mapping input to last conv layer activations and predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Record operations for gradient computation
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
        
    # Get gradients of the score for class_channel wrt last conv layer activations
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Mean intensity of the gradient over each feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply channels by gradients and sum
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
        
    return heatmap.numpy()

def generate_and_save_gradcam(letter_image_gray, model, save_path, pred_index):
    """
    Generates and saves the colorized Grad-CAM heatmap overlaid on the letter image.
    
    Args:
        letter_image_gray: Grayscale character crop (un-normalized).
        model: The loaded Keras model.
        save_path: Absolute filepath to save the output PNG.
        pred_index: Predicted class index (0, 1, or 2).
    """
    # 1. Preprocess the character crop for prediction
    resized = cv2.resize(letter_image_gray, (28, 28))
    # Duplicate grayscale channel to match model's expected 3-channel input shape
    bgr = cv2.merge([resized, resized, resized])
    normalized = bgr / 255.0
    img_array = np.expand_dims(normalized, axis=0) # Shape: (1, 28, 28, 3)
    
    # 2. Get Grad-CAM Heatmap
    try:
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_5", pred_index=pred_index)
        
        # 3. Resize heatmap to match original character image dimensions
        h, w = letter_image_gray.shape
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # 4. Convert heatmap to RGB Jet Colormap
        heatmap_color = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
        
        # 5. Overlay onto character BGR image
        char_bgr = cv2.merge([letter_image_gray, letter_image_gray, letter_image_gray])
        overlay = cv2.addWeighted(char_bgr, 0.6, heatmap_color, 0.4, 0)
        
        # 6. Save image to disk
        cv2.imwrite(save_path, overlay)
        return True
    except Exception as e:
        print(f"Grad-CAM Generation Error: {e}")
        # Fallback: Save original image in case of error
        char_bgr = cv2.merge([letter_image_gray, letter_image_gray, letter_image_gray])
        cv2.imwrite(save_path, char_bgr)
        return False
