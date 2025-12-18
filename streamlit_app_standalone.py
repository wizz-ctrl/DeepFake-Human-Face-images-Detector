"""
Deepfake Detection - Standalone Streamlit Application
All inference is done locally - no external backend needed
Supports both CNN (EfficientNetV2B0) and SVM models
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import joblib
import os
import sys
from scipy.fftpack import dct, fft2

# Add SVM_model folder to path for imports
SVM_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SVM_model')
sys.path.insert(0, SVM_MODEL_DIR)

# ============================================
# CONFIGURATION
# ============================================

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# CNN Model Path (EfficientNetV2B0)
CNN_MODEL_PATH = os.path.join(SCRIPT_DIR, 'deepfake_detector_v4_40k.h5')

# SVM Model Paths
SVM_MODEL_PATH = os.path.join(SCRIPT_DIR, 'SVM_model', 'fake_face_detector.pkl')
SVM_SCALER_PATH = os.path.join(SCRIPT_DIR, 'SVM_model', 'scaler.pkl')
SVM_SELECTOR_PATH = os.path.join(SCRIPT_DIR, 'SVM_model', 'feature_selector.pkl')

# DNN Face Detector paths
DNN_PROTOTXT = os.path.join(SCRIPT_DIR, 'deploy.prototxt')
DNN_CAFFEMODEL = os.path.join(SCRIPT_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')

# Legacy compatibility
MODEL_PATH = CNN_MODEL_PATH

# Fallback to parent directory for CNN model
if not os.path.exists(CNN_MODEL_PATH):
    CNN_MODEL_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'deepfake_detector_v4_40k.h5')
    MODEL_PATH = CNN_MODEL_PATH

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Deepfake Detector",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM STYLING
# ============================================

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Audiowide&family=Exo+2:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    /* Background gradient - dark blue (left) to light blue (right) */
    .stApp {
        background: linear-gradient(to right, #0a1628, #1a365d, #2563eb, #60a5fa, #bfdbfe) !important;
    }
    
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #0a1628, #1e3a5f) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #e0e7ff !important;
    }
    
    /* Main title - Magneto-like style (Audiowide is similar) */
    .main-header {
        font-family: 'Audiowide', cursive !important;
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* All other text - Daytona-like style (Exo 2 is similar) */
    html, body, [class*="css"] {
        font-family: 'Exo 2', sans-serif !important;
    }
    
    .sub-header {
        font-family: 'Exo 2', sans-serif !important;
        font-size: 1.2rem;
        color: #e0e7ff;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Force font on all Streamlit elements */
    .stApp, .stSidebar, div, p, span, label, button, input, textarea, 
    .stMarkdown, .stText, h1, h2, h3, h4, h5, h6,
    .stMetric, .stExpander, .stSelectbox, .stMultiSelect,
    [data-testid="stSidebar"], [data-testid="stHeader"],
    .element-container, .stAlert, .stSuccess, .stError, .stWarning, .stInfo {
        font-family: 'Exo 2', sans-serif !important;
    }
    
    /* Text colors for readability on gradient */
    .stMarkdown, .stText, p, h2, h3, h4, h5, h6, label, span {
        color: #ffffff !important;
    }
    
    /* Make metrics readable */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #ffffff !important;
    }
    
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .real-card {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-left: 5px solid #4CAF50;
    }
    .fake-card {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        border-left: 5px solid #F44336;
    }
    .metric-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .stProgress > div > div > div { background-color: #1E88E5; }
    .model-selector {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# SVM FEATURE EXTRACTION (from train_model.py)
# ============================================

try:
    from skimage.feature import hog
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

def extract_svm_features(img):
    """Extract features from an image for SVM model (same as train_model.py)"""
    try:
        # Resize to 160x160 as expected by SVM model
        img = cv2.resize(img, (160, 160))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Convert RGB to BGR for OpenCV operations
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        features = []
        
        # 1. Color histogram features (32 bins per channel)
        for i in range(3):
            hist = cv2.calcHist([img_bgr], [i], None, [32], [0, 256])
            features.extend(hist.flatten())
        
        # 2. LBP features with multiple radii
        for radius in [1, 2]:
            padded = np.pad(gray, radius, mode='edge')
            center = gray.astype(np.int32)
            
            lbp = np.zeros_like(gray, dtype=np.int32)
            if radius == 1:
                lbp += ((padded[0:-2, 0:-2] >= center).astype(np.int32)) * 128
                lbp += ((padded[0:-2, 1:-1] >= center).astype(np.int32)) * 64
                lbp += ((padded[0:-2, 2:] >= center).astype(np.int32)) * 32
                lbp += ((padded[1:-1, 2:] >= center).astype(np.int32)) * 16
                lbp += ((padded[2:, 2:] >= center).astype(np.int32)) * 8
                lbp += ((padded[2:, 1:-1] >= center).astype(np.int32)) * 4
                lbp += ((padded[2:, 0:-2] >= center).astype(np.int32)) * 2
                lbp += ((padded[1:-1, 0:-2] >= center).astype(np.int32)) * 1
            else:
                r = radius
                lbp += ((padded[0:gray.shape[0], 0:gray.shape[1]] >= center).astype(np.int32)) * 128
                lbp += ((padded[0:gray.shape[0], r:gray.shape[1]+r] >= center).astype(np.int32)) * 64
                lbp += ((padded[0:gray.shape[0], 2*r:gray.shape[1]+2*r] >= center).astype(np.int32)) * 32
                lbp += ((padded[r:gray.shape[0]+r, 2*r:gray.shape[1]+2*r] >= center).astype(np.int32)) * 16
                lbp += ((padded[2*r:gray.shape[0]+2*r, 2*r:gray.shape[1]+2*r] >= center).astype(np.int32)) * 8
                lbp += ((padded[2*r:gray.shape[0]+2*r, r:gray.shape[1]+r] >= center).astype(np.int32)) * 4
                lbp += ((padded[2*r:gray.shape[0]+2*r, 0:gray.shape[1]] >= center).astype(np.int32)) * 2
                lbp += ((padded[r:gray.shape[0]+r, 0:gray.shape[1]] >= center).astype(np.int32)) * 1
            
            lbp = lbp.astype(np.uint8)
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=32, range=(0, 256))
            features.extend(lbp_hist)
        
        # 3. Edge detection
        edges_canny = cv2.Canny(gray, 100, 200)
        edge_density_canny = np.sum(edges_canny) / (edges_canny.shape[0] * edges_canny.shape[1])
        features.append(edge_density_canny)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.extend([np.mean(np.abs(laplacian)), np.std(laplacian)])
        
        # 4. DCT features
        dct_img = dct(dct(gray.T, norm='ortho').T, norm='ortho')
        dct_features = dct_img[:24, :24].flatten()
        features.extend(dct_features)
        
        # 5. Statistical features
        features.extend([
            np.mean(gray), np.std(gray), np.median(gray),
            np.percentile(gray, 25), np.percentile(gray, 75),
            np.min(gray), np.max(gray),
            np.percentile(gray, 10), np.percentile(gray, 90)
        ])
        
        # 6. Gradient features
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        gradient_direction = np.arctan2(gy, gx)
        
        features.extend([
            np.mean(gradient_magnitude), np.std(gradient_magnitude),
            np.max(gradient_magnitude), np.median(gradient_magnitude)
        ])
        
        dir_hist, _ = np.histogram(gradient_direction, bins=16, range=(-np.pi, np.pi))
        features.extend(dir_hist)
        
        # 7. Color channel statistics
        for channel in cv2.split(img_bgr):
            features.extend([np.mean(channel), np.std(channel)])
        
        # 8. GLCM-inspired features
        h_diff = np.abs(gray[:, 1:].astype(np.int32) - gray[:, :-1].astype(np.int32))
        features.extend([np.mean(h_diff), np.std(h_diff)])
        
        v_diff = np.abs(gray[1:, :].astype(np.int32) - gray[:-1, :].astype(np.int32))
        features.extend([np.mean(v_diff), np.std(v_diff)])
        
        # 9. HOG features
        if SKIMAGE_AVAILABLE:
            hog_features = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                              cells_per_block=(2, 2), visualize=False)
            features.extend(hog_features[:200])
        
        # 10. FFT features
        fft_img = fft2(gray)
        fft_shifted = np.fft.fftshift(fft_img)
        magnitude_spectrum = np.abs(fft_shifted)
        
        h, w = magnitude_spectrum.shape
        center_region = magnitude_spectrum[h//4:3*h//4, w//4:3*w//4]
        features.extend([
            np.mean(magnitude_spectrum), np.std(magnitude_spectrum),
            np.mean(center_region), np.std(center_region),
            np.max(magnitude_spectrum), np.median(magnitude_spectrum)
        ])
        
        # 11. HSV features
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        for channel in range(3):
            hist = cv2.calcHist([hsv], [channel], None, [16], [0, 256])
            features.extend(hist.flatten())
        
        # 12. LAB features
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        for channel in range(3):
            features.extend([np.mean(lab[:,:,channel]), np.std(lab[:,:,channel])])
        
        # 13. Noise estimation
        high_pass = gray.astype(np.float32) - cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
        features.extend([np.mean(np.abs(high_pass)), np.std(high_pass)])
        
        # 14. Blur detection
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features.append(laplacian_var)
        
        return np.array(features)
    except Exception as e:
        st.error(f"Error extracting SVM features: {e}")
        return None


# ============================================
# MODEL LOADING
# ============================================

@st.cache_resource
def load_cnn_model():
    """Load the CNN (EfficientNetV2B0) deepfake detection model."""
    if not os.path.exists(CNN_MODEL_PATH):
        return None
    
    try:
        model = tf.keras.models.load_model(CNN_MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading CNN model: {str(e)}")
        return None


@st.cache_resource
def load_svm_model():
    """Load the SVM model with scaler and feature selector."""
    svm_components = {'model': None, 'scaler': None, 'selector': None}
    
    if not os.path.exists(SVM_MODEL_PATH):
        return svm_components
    
    try:
        svm_components['model'] = joblib.load(SVM_MODEL_PATH)
        
        if os.path.exists(SVM_SCALER_PATH):
            svm_components['scaler'] = joblib.load(SVM_SCALER_PATH)
        
        if os.path.exists(SVM_SELECTOR_PATH):
            svm_components['selector'] = joblib.load(SVM_SELECTOR_PATH)
        
        return svm_components
    except Exception as e:
        st.error(f"Error loading SVM model: {str(e)}")
        return svm_components


@st.cache_resource
def load_face_detector():
    """Load the DNN face detector."""
    if os.path.exists(DNN_PROTOTXT) and os.path.exists(DNN_CAFFEMODEL):
        try:
            face_net = cv2.dnn.readNetFromCaffe(DNN_PROTOTXT, DNN_CAFFEMODEL)
            return face_net
        except Exception as e:
            st.error(f"Error loading face detector: {str(e)}")
            return None
    return None


@st.cache_resource
def load_model():
    """Load the deepfake detection model and DNN face detector (legacy function)."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        return None, None
    
    try:
        # Load deepfake detection model
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Load DNN face detector (much more accurate than Haar cascades)
        if os.path.exists(DNN_PROTOTXT) and os.path.exists(DNN_CAFFEMODEL):
            face_net = cv2.dnn.readNetFromCaffe(DNN_PROTOTXT, DNN_CAFFEMODEL)
        else:
            st.warning("DNN face detector not found, face detection may be less accurate")
            face_net = None
        
        return model, face_net
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


# ============================================
# FACE DETECTION (DNN-based - More Accurate)
# ============================================

def boxes_overlap_iou(box1, box2):
    """Calculate IoU (Intersection over Union) between two boxes."""
    x1, y1, w1, h1 = box1[:4]
    x2, y2, w2, h2 = box2[:4]
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def is_contained_in(small_box, large_box, threshold=0.7):
    """
    Check if small_box is mostly contained within large_box.
    Returns True if 'threshold' fraction of small_box area is inside large_box.
    This catches nested face detections (face inside face).
    """
    x1, y1, w1, h1 = small_box[:4]
    x2, y2, w2, h2 = large_box[:4]
    
    # Calculate intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return False
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    small_area = w1 * h1
    
    # Check if most of small box is inside large box
    if small_area > 0:
        containment_ratio = intersection / small_area
        return containment_ratio >= threshold
    
    return False


def detect_faces_single_pass(bgr_image, face_net, confidence_threshold=0.3):
    """Run a single detection pass on an image."""
    h, w = bgr_image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(
        bgr_image,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
        swapRB=False,
        crop=False
    )
    
    face_net.setInput(blob)
    detections = face_net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            face_w = x2 - x1
            face_h = y2 - y1
            
            if face_w > 10 and face_h > 10:
                faces.append((x1, y1, face_w, face_h, float(confidence)))
    
    return faces


def detect_faces_dnn(image_array, face_net, confidence_threshold=0.3):
    """
    Detect faces using OpenCV's DNN module with sliding window for small faces.
    Uses tiling to detect small distant faces that get lost in full-image detection.
    
    Args:
        image_array: RGB numpy array
        face_net: OpenCV DNN network
        confidence_threshold: Minimum confidence to accept a detection (0.0-1.0)
    
    Returns:
        List of (x, y, w, h) tuples for detected faces
    """
    if face_net is None:
        return []
    
    h, w = image_array.shape[:2]
    all_faces = []
    
    # Convert to BGR once
    bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # ===== PASS 1: Full image detection (for large faces) =====
    full_faces = detect_faces_single_pass(bgr_image, face_net, confidence_threshold)
    for (x1, y1, fw, fh, conf) in full_faces:
        all_faces.append((x1, y1, fw, fh, conf))
    
    # ===== PASS 2: Tiled detection (for small faces) =====
    # Split image into overlapping tiles and detect in each
    # This helps because small faces become larger relative to the 300x300 input
    
    tile_sizes = []
    
    # Only tile if image is large enough
    if w > 600 and h > 600:
        tile_sizes.append((2, 2))  # 2x2 grid (4 tiles)
    if w > 900 and h > 900:
        tile_sizes.append((3, 3))  # 3x3 grid (9 tiles)
    
    for (grid_x, grid_y) in tile_sizes:
        tile_w = w // grid_x
        tile_h = h // grid_y
        overlap = 0.25  # 25% overlap between tiles
        
        step_x = int(tile_w * (1 - overlap))
        step_y = int(tile_h * (1 - overlap))
        
        for ty in range(0, h - tile_h + 1, step_y):
            for tx in range(0, w - tile_w + 1, step_x):
                # Extract tile
                tile = bgr_image[ty:ty+tile_h, tx:tx+tile_w]
                
                # Detect faces in tile
                tile_faces = detect_faces_single_pass(tile, face_net, confidence_threshold)
                
                # Convert tile coordinates back to full image coordinates
                for (x1, y1, fw, fh, conf) in tile_faces:
                    # Map back to original image
                    abs_x = tx + x1
                    abs_y = ty + y1
                    all_faces.append((abs_x, abs_y, fw, fh, conf))
    
    # ===== PASS 3: Upscaled detection (another approach for small faces) =====
    if max(h, w) < 1200:
        scale = 2.0
        upscaled = cv2.resize(bgr_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        up_faces = detect_faces_single_pass(upscaled, face_net, confidence_threshold)
        
        for (x1, y1, fw, fh, conf) in up_faces:
            # Scale back to original coordinates
            abs_x = int(x1 / scale)
            abs_y = int(y1 / scale)
            abs_w = int(fw / scale)
            abs_h = int(fh / scale)
            if abs_w > 10 and abs_h > 10:
                all_faces.append((abs_x, abs_y, abs_w, abs_h, conf))
    
    # ===== Non-Maximum Suppression =====
    # Sort by area (largest first) - prefer larger face detections
    all_faces.sort(key=lambda f: f[2] * f[3], reverse=True)
    
    # NMS: keep only non-overlapping faces, suppress nested faces
    final_faces = []
    for face in all_faces:
        is_duplicate = False
        face_area = face[2] * face[3]
        
        for existing in final_faces:
            existing_area = existing[2] * existing[3]
            
            # Check IoU overlap
            if boxes_overlap_iou(face, existing) > 0.3:
                is_duplicate = True
                break
            
            # Check if current face is inside an existing larger face (nested detection)
            if face_area < existing_area and is_contained_in(face, existing, threshold=0.6):
                is_duplicate = True
                break
            
            # Check if an existing face would be inside this face (shouldn't happen since sorted by area)
            if existing_area < face_area and is_contained_in(existing, face, threshold=0.6):
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_faces.append(face)
    
    # Return without confidence score for compatibility
    return [(x, y, w, h) for x, y, w, h, _ in final_faces]


def extract_face(image_array, bbox, padding_percent=0.25):
    """Extract and crop a face region with padding."""
    x, y, w, h = bbox
    img_h, img_w = image_array.shape[:2]
    
    pad_w = int(w * padding_percent)
    pad_h = int(h * padding_percent)
    
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)
    
    face_crop = image_array[y1:y2, x1:x2]
    
    if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
        return None
    
    face_resized = cv2.resize(face_crop, (256, 256), interpolation=cv2.INTER_AREA)
    return face_resized


def classify_face_cnn(face_array, model):
    """Run CNN model inference on a face."""
    if len(face_array.shape) == 2:
        face_array = cv2.cvtColor(face_array, cv2.COLOR_GRAY2RGB)
    elif face_array.shape[2] == 4:
        face_array = cv2.cvtColor(face_array, cv2.COLOR_RGBA2RGB)
    
    batch = np.expand_dims(face_array, axis=0).astype(np.float32)
    prediction = model.predict(batch, verbose=0)[0][0]
    
    is_fake = prediction >= 0.5
    confidence = float(prediction) if is_fake else float(1 - prediction)
    label = "FAKE" if is_fake else "REAL"
    
    return label, confidence, float(prediction)


def classify_face_svm(face_array, svm_components):
    """Run SVM model inference on a face."""
    if svm_components['model'] is None:
        return "UNKNOWN", 0.0, 0.5
    
    if len(face_array.shape) == 2:
        face_array = cv2.cvtColor(face_array, cv2.COLOR_GRAY2RGB)
    elif face_array.shape[2] == 4:
        face_array = cv2.cvtColor(face_array, cv2.COLOR_RGBA2RGB)
    
    # Extract features for SVM
    features = extract_svm_features(face_array)
    
    if features is None:
        return "UNKNOWN", 0.0, 0.5
    
    try:
        # Scale features
        if svm_components['scaler'] is not None:
            features_scaled = svm_components['scaler'].transform(features.reshape(1, -1))
        else:
            features_scaled = features.reshape(1, -1)
        
        # Apply feature selection
        if svm_components['selector'] is not None:
            features_scaled = svm_components['selector'].transform(features_scaled)
        
        # Predict
        prediction = svm_components['model'].predict(features_scaled)[0]
        probability = svm_components['model'].predict_proba(features_scaled)[0]
        
        # SVM: 0 = Real, 1 = Fake
        is_fake = prediction == 1
        raw_score = probability[1]  # Probability of being fake
        confidence = probability[prediction]
        label = "FAKE" if is_fake else "REAL"
        
        return label, float(confidence), float(raw_score)
    except Exception as e:
        st.error(f"SVM prediction error: {e}")
        return "UNKNOWN", 0.0, 0.5


def classify_face(face_array, model, model_type="cnn", svm_components=None):
    """Run model inference on a face - wrapper for both models."""
    if model_type == "svm":
        return classify_face_svm(face_array, svm_components)
    else:
        return classify_face_cnn(face_array, model)


def analyze_image(image, model, face_net, model_type="cnn", svm_components=None):
    """Main analysis pipeline using DNN face detection."""
    # Convert PIL to numpy
    image_array = np.array(image)
    
    # Ensure RGB
    if len(image_array.shape) == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.shape[2] == 4:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    
    # Detect faces using DNN (much more accurate)
    face_boxes = detect_faces_dnn(image_array, face_net, confidence_threshold=0.5)
    
    if len(face_boxes) == 0:
        return {
            "success": True,
            "faces": [],
            "summary": {
                "total_faces": 0,
                "real_count": 0,
                "fake_count": 0,
                "overall_verdict": "NO FACES DETECTED"
            },
            "annotated_image": image_array
        }
    
    # Process each face
    results = []
    annotated = image_array.copy()
    real_count = 0
    fake_count = 0
    
    for i, bbox in enumerate(face_boxes):
        x, y, w, h = [int(v) for v in bbox]
        face_crop = extract_face(image_array, bbox)
        
        if face_crop is None:
            continue
        
        label, confidence, raw_score = classify_face(face_crop, model, model_type, svm_components)
        
        if label == "REAL":
            real_count += 1
            color = (0, 255, 0)  # Green
        else:
            fake_count += 1
            color = (255, 0, 0)  # Red
        
        # Draw on annotated image
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        text = f"{label} ({confidence*100:.1f}%)"
        cv2.putText(annotated, text, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        results.append({
            "face_id": i + 1,
            "bbox": {"x": x, "y": y, "width": w, "height": h},
            "label": label,
            "confidence": confidence,
            "raw_score": raw_score,
            "face_image": Image.fromarray(face_crop)
        })
    
    total = len(results)
    
    # Calculate aggregated statistics for verdict
    if total > 0:
        fake_percentage = round((fake_count / total) * 100, 1)
        real_percentage = round((real_count / total) * 100, 1)
        
        # Calculate average confidence scores
        fake_confidences = [f['confidence'] for f in results if f['label'] == 'FAKE']
        real_confidences = [f['confidence'] for f in results if f['label'] == 'REAL']
        avg_fake_confidence = sum(fake_confidences) / len(fake_confidences) if fake_confidences else 0
        avg_real_confidence = sum(real_confidences) / len(real_confidences) if real_confidences else 0
        
        # Calculate weighted fake score (considers both count and confidence)
        weighted_fake_score = (fake_count / total) * avg_fake_confidence if fake_count > 0 else 0
        
        # Determine overall verdict with reasoning
        if fake_count == 0:
            overall_verdict = "AUTHENTIC"
            verdict_confidence = "HIGH"
            verdict_explanation = "All detected faces appear to be genuine with no signs of manipulation."
        elif fake_count == total:
            overall_verdict = "FAKE"
            verdict_confidence = "HIGH"
            verdict_explanation = "All detected faces show signs of AI manipulation or deepfake generation."
        elif fake_percentage >= 50:
            overall_verdict = "LIKELY FAKE"
            verdict_confidence = "MEDIUM" if avg_fake_confidence < 0.8 else "HIGH"
            verdict_explanation = f"Majority of faces ({fake_percentage}%) appear to be manipulated."
        elif fake_percentage >= 25:
            overall_verdict = "SUSPICIOUS"
            verdict_confidence = "MEDIUM"
            verdict_explanation = f"Some faces ({fake_percentage}%) show signs of manipulation. Manual review recommended."
        else:
            overall_verdict = "MOSTLY AUTHENTIC"
            verdict_confidence = "MEDIUM"
            verdict_explanation = f"Most faces appear genuine, but {fake_count} face(s) may be manipulated."
    else:
        fake_percentage = 0
        real_percentage = 0
        avg_fake_confidence = 0
        avg_real_confidence = 0
        weighted_fake_score = 0
        overall_verdict = "NO FACES DETECTED"
        verdict_confidence = "N/A"
        verdict_explanation = "No faces were detected in the image."
    
    summary = {
        "total_faces": total,
        "real_count": real_count,
        "fake_count": fake_count,
        "fake_percentage": fake_percentage,
        "real_percentage": real_percentage,
        "avg_fake_confidence": round(avg_fake_confidence * 100, 1),
        "avg_real_confidence": round(avg_real_confidence * 100, 1),
        "weighted_fake_score": round(weighted_fake_score * 100, 1),
        "overall_verdict": overall_verdict,
        "verdict_confidence": verdict_confidence,
        "verdict_explanation": verdict_explanation
    }
    
    return {
        "success": True,
        "faces": results,
        "summary": summary,
        "annotated_image": annotated
    }


# ============================================
# MAIN APP
# ============================================

def main():
    # Header
    st.markdown('<h1 class="main-header">Deepfake Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Face Authentication</p>', unsafe_allow_html=True)
    
    # Load all models
    cnn_model = load_cnn_model()
    svm_components = load_svm_model()
    face_net = load_face_detector()
    
    # Sidebar with Model Selection
    with st.sidebar:
        st.markdown("## 🎯 Select Detection Model")
        
        # Model selector
        model_choice = st.radio(
            "Choose Model:",
            ["CNN (EfficientNetV2B0)", "SVM (Machine Learning)"],
            index=0,
            help="Select which model to use for deepfake detection"
        )
        
        st.markdown("---")
        
        # Show model-specific info based on selection
        if model_choice == "CNN (EfficientNetV2B0)":
            st.markdown("## 🧠 CNN Model Info")
            st.markdown("""
            **Architecture:** EfficientNetV2B0
            
            **Performance Metrics:**
            - ✅ Accuracy: **96.5%**
            - 🎯 Fake Detection: 94.7%
            - ✅ Real Detection: 98.3%
            - 📈 AUC Score: 99.39%
            
            **Features:**
            - Deep Learning based
            - Pre-trained on ImageNet
            - 40,000 training images
            """)
            
            # Check model status
            if cnn_model is not None:
                st.success("✓ CNN Model loaded")
            else:
                st.error("✗ CNN Model not found")
        else:
            st.markdown("## 🔬 SVM Model Info")
            st.markdown("""
            **Architecture:** Random Forest + SVM Ensemble
            
            **Performance Metrics:**
            - ✅ Accuracy: **82.25%**
            
            **Features Used:**
            - Color histograms (RGB, HSV, LAB)
            - LBP texture patterns
            - HOG features
            - DCT/FFT frequency analysis
            - Edge & gradient features
            - Statistical features
            """)
            
            # Check model status
            if svm_components['model'] is not None:
                st.success("✓ SVM Model loaded")
                if svm_components['scaler'] is not None:
                    st.success("✓ Scaler loaded")
                if svm_components['selector'] is not None:
                    st.success("✓ Feature selector loaded")
            else:
                st.error("✗ SVM Model not found")
        
        st.markdown("---")
        st.markdown("## How It Works")
        st.markdown("""
        1. **Select** a detection model
        2. **Upload** any image with faces
        3. **Detect** all faces automatically
        4. **Analyze** each face for deepfakes
        5. **Review** detailed results
        """)
    
    # Determine which model to use
    model_type = "cnn" if model_choice == "CNN (EfficientNetV2B0)" else "svm"
    active_model = cnn_model if model_type == "cnn" else svm_components['model']
    
    # Check if selected model is available
    if model_type == "cnn" and cnn_model is None:
        st.error("❌ CNN model not found. Please check the model file path.")
        st.info(f"Expected at: {CNN_MODEL_PATH}")
        return
    
    if model_type == "svm" and svm_components['model'] is None:
        st.error("❌ SVM model not found. Please check the model file path.")
        st.info(f"Expected at: {SVM_MODEL_PATH}")
        return
    
    if face_net is None:
        st.warning("⚠️ Face detector not loaded. Please ensure deploy.prototxt and .caffemodel files are present.")
        return
    
    # Show selected model
    st.success(f"✅ Using **{model_choice}** for detection")
    
    # File uploader
    st.markdown("### Upload Image")
    uploaded_file = st.file_uploader(
        "Drag and drop file here",
        type=["jpg", "jpeg", "png", "webp"],
        help="Limit 200MB per file • JPG, JPEG, PNG, WEBP"
    )
    
    if uploaded_file is not None:
        # Display original image FIRST (before analysis)
        image = Image.open(uploaded_file)
        
        # Show original image immediately
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("### Detected Faces")
            # Placeholder for analysis result
            analysis_placeholder = st.empty()
            analysis_placeholder.info(f"Analyzing with {model_choice}...")
        
        # Now run the analysis with selected model
        with st.spinner(f"Processing with {model_choice}..."):
            result = analyze_image(image, active_model, face_net, model_type, svm_components)
        
        # Update the right column with results
        with col2:
            analysis_placeholder.empty()
            if result["success"] and result["summary"]["total_faces"] > 0:
                st.image(result["annotated_image"], use_container_width=True)
            else:
                st.image(image, use_container_width=True)
                st.caption("No faces detected")
        
        # Full-width section for analysis results
        st.markdown("---")
        
        if result["success"]:
            summary = result["summary"]
            
            if summary["total_faces"] == 0:
                st.warning("No human faces detected in the image. Please upload an image containing human faces.")
            else:
                # Summary metrics (full width)
                st.markdown("### Analysis Summary")
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Faces", summary["total_faces"])
                m2.metric("Real Faces", summary["real_count"], f"{summary['real_percentage']}%")
                m3.metric("Fake Faces", summary["fake_count"], f"{summary['fake_percentage']}%")
                
                # ===== OVERALL IMAGE VERDICT (full width) =====
                st.markdown("---")
                st.markdown("### Overall Image Verdict")
                
                verdict = summary["overall_verdict"]
                confidence = summary["verdict_confidence"]
                explanation = summary["verdict_explanation"]
                
                # Verdict card with color coding
                if verdict == "AUTHENTIC":
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%); 
                                padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">
                        <h2 style="color: #2E7D32; margin: 0;">AUTHENTIC</h2>
                        <p style="color: #1B5E20; font-size: 16px; margin-top: 10px;">{explanation}</p>
                        <p style="color: #388E3C;"><strong>Confidence:</strong> {confidence}</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif verdict == "FAKE":
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%); 
                                padding: 20px; border-radius: 10px; border-left: 5px solid #F44336;">
                        <h2 style="color: #C62828; margin: 0;">FAKE / DEEPFAKE</h2>
                        <p style="color: #B71C1C; font-size: 16px; margin-top: 10px;">{explanation}</p>
                        <p style="color: #D32F2F;"><strong>Confidence:</strong> {confidence}</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif verdict == "LIKELY FAKE":
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%); 
                                padding: 20px; border-radius: 10px; border-left: 5px solid #FF9800;">
                        <h2 style="color: #E65100; margin: 0;">LIKELY FAKE</h2>
                        <p style="color: #EF6C00; font-size: 16px; margin-top: 10px;">{explanation}</p>
                        <p style="color: #F57C00;"><strong>Confidence:</strong> {confidence}</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif verdict == "SUSPICIOUS":
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #FFF8E1 0%, #FFECB3 100%); 
                                padding: 20px; border-radius: 10px; border-left: 5px solid #FFC107;">
                        <h2 style="color: #FF8F00; margin: 0;">SUSPICIOUS</h2>
                        <p style="color: #FF6F00; font-size: 16px; margin-top: 10px;">{explanation}</p>
                        <p style="color: #FFA000;"><strong>Confidence:</strong> {confidence}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:  # MOSTLY AUTHENTIC
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%); 
                                padding: 20px; border-radius: 10px; border-left: 5px solid #2196F3;">
                        <h2 style="color: #1565C0; margin: 0;">MOSTLY AUTHENTIC</h2>
                        <p style="color: #0D47A1; font-size: 16px; margin-top: 10px;">{explanation}</p>
                        <p style="color: #1976D2;"><strong>Confidence:</strong> {confidence}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Additional stats (full width, side by side)
                if summary["fake_count"] > 0 or summary["real_count"] > 0:
                    st.markdown("<br>", unsafe_allow_html=True)
                    stat1, stat2 = st.columns(2)
                    with stat1:
                        if summary["avg_real_confidence"] > 0:
                            st.info(f"Avg. Real Face Confidence: **{summary['avg_real_confidence']}%**")
                    with stat2:
                        if summary["avg_fake_confidence"] > 0:
                            st.warning(f"Avg. Fake Face Confidence: **{summary['avg_fake_confidence']}%**")
                
                # Individual face results (full width)
                if len(result["faces"]) > 0:
                    st.markdown("---")
                    st.markdown("### Individual Face Analysis")
                    
                    # Display faces in a grid layout with smaller images
                    num_faces = len(result["faces"])
                    cols_per_row = min(4, num_faces)  # Max 4 faces per row for smaller size
                    
                    for i in range(0, num_faces, cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            if i + j < num_faces:
                                face = result["faces"][i + j]
                                with col:
                                    # Use fixed width for smaller face images
                                    st.image(face["face_image"], caption=f"Face #{face['face_id']}", width=120)
                                    if face["label"] == "REAL":
                                        st.success(f"**{face['label']}** ({face['confidence']*100:.1f}%)")
                                    else:
                                        st.error(f"**{face['label']}** ({face['confidence']*100:.1f}%)")
                                    st.progress(face['confidence'])
        else:
            st.error("Analysis failed. Please try again with a different image.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #666;">
        <p style="font-size: 18px; margin-bottom: 10px;">Made with ❤️ by <strong>Taimoor</strong> and <strong>Ahmed</strong> for 🌍</p>
        <p style="font-size: 12px; color: #999;">© 2025 Deepfake Detection System. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
