import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
import tempfile
import os

# Import feature extraction from train_model
from train_model import extract_features

# Page configuration
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling (matching the template)
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --bg-dark: #0a0a0f;
        --bg-card: #12121a;
        --border-color: #2a2a3a;
        --text-primary: #ffffff;
        --text-secondary: #a0a0b0;
        --accent-purple: #8b5cf6;
        --success-green: #10b981;
        --danger-red: #ef4444;
    }
    
    /* Dark theme */
    .stApp {
        background-color: #0a0a0f;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #12121a;
        border-right: 1px solid #2a2a3a;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #a0a0b0;
    }
    
    /* Card styling */
    .card {
        background: #12121a;
        border: 1px solid #2a2a3a;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }
    
    .card-header {
        font-size: 18px;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid #2a2a3a;
    }
    
    /* Model info card */
    .model-card {
        background: #0a0a0f;
        border: 1px solid #2a2a3a;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .model-status {
        display: inline-block;
        font-size: 11px;
        padding: 4px 8px;
        border-radius: 6px;
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
    }
    
    /* Metric cards */
    .metric-card {
        background: #12121a;
        border: 1px solid #2a2a3a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    
    .metric-label {
        color: #a0a0b0;
        font-size: 14px;
    }
    
    .metric-green { color: #10b981; }
    .metric-red { color: #ef4444; }
    .metric-purple { color: #8b5cf6; }
    
    /* Verdict cards */
    .verdict-authentic {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), transparent);
        border: 1px solid #10b981;
        border-radius: 16px;
        padding: 24px;
    }
    
    .verdict-fake {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), transparent);
        border: 1px solid #ef4444;
        border-radius: 16px;
        padding: 24px;
    }
    
    .verdict-title-green {
        color: #10b981;
        font-size: 24px;
        font-weight: 700;
    }
    
    .verdict-title-red {
        color: #ef4444;
        font-size: 24px;
        font-weight: 700;
    }
    
    /* Progress bars */
    .progress-container {
        background: #0a0a0f;
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
        margin-top: 8px;
    }
    
    .progress-green {
        background: linear-gradient(90deg, #10b981, #34d399);
        height: 100%;
        border-radius: 4px;
    }
    
    .progress-red {
        background: linear-gradient(90deg, #ef4444, #f87171);
        height: 100%;
        border-radius: 4px;
    }
    
    /* Upload area */
    .upload-zone {
        border: 2px dashed #2a2a3a;
        border-radius: 12px;
        padding: 48px;
        text-align: center;
        background: #12121a;
        transition: all 0.3s ease;
    }
    
    .upload-icon {
        font-size: 48px;
        margin-bottom: 16px;
    }
    
    /* Header */
    .main-header {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }
    
    .main-subtitle {
        color: #a0a0b0;
        font-size: 15px;
        margin-bottom: 32px;
    }
    
    /* Logo */
    .logo-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 32px;
    }
    
    .logo-icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #8b5cf6, #7c3aed);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }
    
    .logo-text {
        font-size: 18px;
        font-weight: 700;
        color: #ffffff;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 32px;
        color: #a0a0b0;
        font-size: 13px;
        border-top: 1px solid #2a2a3a;
        margin-top: 32px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: #12121a;
        border: 2px dashed #2a2a3a;
        border-radius: 12px;
        padding: 20px;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #8b5cf6;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6, #7c3aed);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4);
    }
    
    /* Image styling */
    .image-container {
        background: #12121a;
        border: 1px solid #2a2a3a;
        border-radius: 16px;
        overflow: hidden;
    }
    
    .image-header {
        padding: 16px 20px;
        border-bottom: 1px solid #2a2a3a;
        font-weight: 600;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('fake_face_detector.pkl')
        scaler = joblib.load('scaler.pkl')
        try:
            feature_selector = joblib.load('feature_selector.pkl')
        except FileNotFoundError:
            feature_selector = None
        try:
            model_metadata = joblib.load('model_metadata.pkl')
        except FileNotFoundError:
            model_metadata = {
                'test_accuracy': 0.78,
                'model_type': 'Ensemble'
            }
        return model, scaler, feature_selector, model_metadata, True
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, False

def predict_image(image_path, model, scaler, feature_selector):
    """Predict if an image is real or fake"""
    features = extract_features(image_path)
    
    if features is None:
        return None, None
    
    # Reshape and scale
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Apply feature selection if available
    if feature_selector is not None:
        features_scaled = feature_selector.transform(features_scaled)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return prediction, probability

def process_image_with_overlay(image_path, prediction):
    """Add prediction overlay to image"""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (400, 400))
    
    label = "FAKE" if prediction == 1 else "REAL"
    color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
    
    cv2.putText(img, f"{label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.rectangle(img, (5, 5), (395, 395), color, 3)
    
    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

# Load models
model, scaler, feature_selector, model_metadata, models_loaded = load_models()

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="logo-container">
        <div class="logo-icon">🔍</div>
        <span class="logo-text">DeepfakeDetector</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Model Information")
    
    if models_loaded:
        accuracy = round(model_metadata.get('test_accuracy', 0.78) * 100, 1)
        model_type = model_metadata.get('model_type', 'Ensemble')
        
        st.markdown(f"""
        <div class="model-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-weight: 600; color: #ffffff;">Ensemble Model</span>
                <span class="model-status">Active</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="color: #a0a0b0;">Accuracy</span>
                <span style="font-weight: 500; color: #ffffff;">{accuracy}%</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #a0a0b0;">Type</span>
                <span style="font-weight: 500; color: #ffffff;">{model_type}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Models not loaded!")
    
    st.markdown("### Features Used")
    st.markdown("""
    <div class="model-card">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="color: #a0a0b0;">LBP Features</span>
            <span style="color: #10b981;">✓</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="color: #a0a0b0;">DCT Analysis</span>
            <span style="color: #10b981;">✓</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="color: #a0a0b0;">FFT Spectrum</span>
            <span style="color: #10b981;">✓</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="color: #a0a0b0;">HOG Features</span>
            <span style="color: #10b981;">✓</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: #a0a0b0;">Color Analysis</span>
            <span style="color: #10b981;">✓</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Instructions")
    st.markdown("""
    <div class="model-card">
        <p style="font-size: 13px; color: #a0a0b0; line-height: 1.6;">
            Upload a face image to detect if it's AI-generated (fake) or authentic (real). 
            Supports JPG, PNG, and WEBP formats.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Main content
st.markdown('<h1 class="main-header">Deepfake Face Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Upload an image to analyze if the face is AI-generated or authentic</p>', unsafe_allow_html=True)

# Upload section
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload Face Image",
    type=['jpg', 'jpeg', 'png', 'webp'],
    help="Drag and drop your image here, or click to browse"
)
st.markdown('</div>', unsafe_allow_html=True)

# Process uploaded image
if uploaded_file is not None and models_loaded:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name
    
    # Show loading
    with st.spinner("🔍 Analyzing image..."):
        # Make prediction
        prediction, probability = predict_image(temp_path, model, scaler, feature_selector)
    
    if prediction is not None:
        # Process image with overlay
        processed_img = process_image_with_overlay(temp_path, prediction)
        
        # Clean up temp file
        os.remove(temp_path)
        
        # Display images side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="image-container">
                <div class="image-header">Original Image</div>
            </div>
            """, unsafe_allow_html=True)
            # Load original image
            original_img = Image.open(uploaded_file)
            st.image(original_img, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="image-container">
                <div class="image-header">Detection Result</div>
            </div>
            """, unsafe_allow_html=True)
            st.image(processed_img, use_container_width=True)
        
        # Metrics
        st.markdown("<br>", unsafe_allow_html=True)
        
        confidence = float(probability[prediction] * 100)
        real_prob = float(probability[0] * 100)
        fake_prob = float(probability[1] * 100)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value metric-purple">{confidence:.1f}%</div>
                <div class="metric-label">Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value metric-green">{real_prob:.1f}%</div>
                <div class="metric-label">Real Probability</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value metric-red">{fake_prob:.1f}%</div>
                <div class="metric-label">Fake Probability</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Verdict and Analysis
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 0:  # Real
                st.markdown(f"""
                <div class="verdict-authentic">
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                        <div style="width: 48px; height: 48px; background: rgba(16, 185, 129, 0.2); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 24px;">✓</div>
                        <div class="verdict-title-green">AUTHENTIC FACE</div>
                    </div>
                    <p style="color: #a0a0b0; font-size: 14px; line-height: 1.6;">
                        This image appears to be an authentic photograph of a real person. 
                        The facial features, skin texture, and lighting patterns are consistent 
                        with genuine photographs.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:  # Fake
                st.markdown(f"""
                <div class="verdict-fake">
                    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                        <div style="width: 48px; height: 48px; background: rgba(239, 68, 68, 0.2); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 24px;">✗</div>
                        <div class="verdict-title-red">AI-GENERATED FACE</div>
                    </div>
                    <p style="color: #a0a0b0; font-size: 14px; line-height: 1.6;">
                        This image appears to be AI-generated or manipulated. 
                        Artifacts in texture patterns, frequency analysis, and 
                        color distributions suggest this is not an authentic photograph.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="card">
                <div class="card-header">Analysis Details</div>
                <div style="padding: 12px 0; border-bottom: 1px solid #2a2a3a;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #a0a0b0;">Real Probability</span>
                        <span style="font-weight: 600; color: #ffffff;">{real_prob:.1f}%</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-green" style="width: {real_prob}%;"></div>
                    </div>
                </div>
                <div style="padding: 12px 0; border-bottom: 1px solid #2a2a3a;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #a0a0b0;">Fake Probability</span>
                        <span style="font-weight: 600; color: #ffffff;">{fake_prob:.1f}%</span>
                    </div>
                    <div class="progress-container">
                        <div class="progress-red" style="width: {fake_prob}%;"></div>
                    </div>
                </div>
                <div style="padding: 12px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: #a0a0b0;">Model Confidence</span>
                        <span style="font-weight: 600; color: #ffffff;">{confidence:.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        os.remove(temp_path)
        st.error("Could not process the image. Please try another image.")

elif not models_loaded:
    st.warning("⚠️ Models not loaded. Please train the model first by running `python train_model.py`")

# Footer
st.markdown("""
<div class="footer">
    <p>Deepfake Detection System • ML Project • Feature Engineering Based Detection</p>
</div>
""", unsafe_allow_html=True)
