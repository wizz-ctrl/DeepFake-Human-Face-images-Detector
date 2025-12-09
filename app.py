import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Page configuration
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('deepfake_detector_v4_40k.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load face detector
@st.cache_resource
def load_face_detector():
    """Load OpenCV's pre-trained face detector"""
    try:
        # Using Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return face_cascade
    except Exception as e:
        st.error(f"Error loading face detector: {e}")
        return None

def detect_and_crop_face(image, padding_percent=0.2):
    """
    Detect face in image and crop it with padding
    Returns cropped face image or original if no face detected
    """
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Convert to RGB if needed
    if len(img_array.shape) == 2:  # Grayscale
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        img_gray = img_array
    elif img_array.shape[2] == 4:  # RGBA
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    else:  # RGB
        img_rgb = img_array
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Load face detector
    face_cascade = load_face_detector()
    if face_cascade is None:
        return image, None, "Face detector not available"
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        return image, None, "No face detected in image"
    
    # Get the largest face (assuming it's the main subject)
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    
    # Add padding around face
    padding_w = int(w * padding_percent)
    padding_h = int(h * padding_percent)
    
    # Calculate crop coordinates with padding (ensure within bounds)
    x1 = max(0, x - padding_w)
    y1 = max(0, y - padding_h)
    x2 = min(img_rgb.shape[1], x + w + padding_w)
    y2 = min(img_rgb.shape[0], y + h + padding_h)
    
    # Crop face region
    face_crop = img_rgb[y1:y2, x1:x2]
    
    # Convert back to PIL Image
    face_pil = Image.fromarray(face_crop)
    
    # Return cropped face and detection info
    detection_info = {
        'faces_found': len(faces),
        'crop_coords': (x1, y1, x2, y2),
        'original_face_coords': (x, y, w, h),
        'crop_size': face_crop.shape[:2]
    }
    
    return face_pil, detection_info, "Face detected and cropped successfully"

def preprocess_image(image, target_size=(256, 256)):
    """Preprocess image for model prediction"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Convert to RGB if needed
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Resize to target size
    img_resized = cv2.resize(img_array, target_size)
    
    # Add batch dimension (model expects batch)
    img_batch = np.expand_dims(img_resized, axis=0)
    
    # Model has built-in Rescaling layer, so pass raw 0-255 values
    return img_batch

def predict_deepfake(model, image):
    """Make prediction on image"""
    try:
        # Preprocess image
        img_processed = preprocess_image(image)
        
        # Get prediction
        prediction = model.predict(img_processed, verbose=0)[0][0]
        
        # Determine result
        is_fake = prediction >= 0.5
        confidence = prediction if is_fake else (1 - prediction)
        
        return {
            'is_fake': is_fake,
            'confidence': float(confidence),
            'raw_score': float(prediction),
            'label': 'FAKE' if is_fake else 'REAL'
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Main app
def main():
    # Header
    st.title("🔍 Deepfake Detector")
    st.markdown("### AI-Powered Image Authentication System")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown("""
        **Performance Metrics:**
        - 🎯 Accuracy: **96.5%**
        - 🚨 Fake Detection: **94.7%**
        - ✅ Real Detection: **98.3%**
        - 📈 AUC: **99.39%**
        
        **Model Details:**
        - Architecture: EfficientNetV2B0
        - Training Data: 40,000 images
        - Input Size: 256×256 pixels
        - Classes: Real vs Fake
        """)
        
        st.markdown("---")
        st.markdown("**How to Use:**")
        st.markdown("""
        1. Upload an image (JPG, PNG, JPEG)
        2. Wait for analysis
        3. View results and confidence score
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image containing a face to check if it's real or deepfake"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Image info
            st.caption(f"📏 Size: {image.size[0]}×{image.size[1]} | Format: {image.format}")
            
            # Detect and crop face
            with st.spinner("Detecting face..."):
                face_image, detection_info, detection_msg = detect_and_crop_face(image)
            
            # Display face detection results
            if detection_info is not None:
                st.success(f"✅ {detection_msg}")
                st.image(face_image, caption="Detected Face (Cropped)", use_column_width=True)
                
                with st.expander("🔍 Face Detection Details"):
                    st.write(f"**Faces Found:** {detection_info['faces_found']}")
                    st.write(f"**Cropped Region Size:** {detection_info['crop_size'][1]}×{detection_info['crop_size'][0]} px")
                    if detection_info['faces_found'] > 1:
                        st.info(f"ℹ️ Multiple faces detected. Using the largest face for analysis.")
            else:
                st.warning(f"⚠️ {detection_msg}")
                st.info("ℹ️ Will analyze the entire image. Results may be less accurate.")
                face_image = image  # Use original image if no face detected
    
    with col2:
        st.subheader("🔬 Analysis Results")
        
        if uploaded_file is not None:
            # Load model
            with st.spinner("Loading model..."):
                model = load_model()
            
            if model is not None:
                # Make prediction on the face-cropped image
                with st.spinner("Analyzing image..."):
                    result = predict_deepfake(model, face_image)
                
                if result:
                    # Display result
                    if result['is_fake']:
                        st.error(f"🚨 **DEEPFAKE DETECTED**")
                        result_color = "#ff4b4b"
                    else:
                        st.success(f"✅ **AUTHENTIC IMAGE**")
                        result_color = "#00c853"
                    
                    # Confidence meter
                    st.markdown("### Confidence Score")
                    confidence_pct = result['confidence'] * 100
                    st.progress(result['confidence'])
                    st.markdown(f"<h2 style='text-align: center; color: {result_color};'>{confidence_pct:.1f}%</h2>", 
                               unsafe_allow_html=True)
                    
                    # Detailed metrics
                    st.markdown("### Detailed Analysis")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Classification", result['label'])
                    with col_b:
                        st.metric("Raw Score", f"{result['raw_score']:.4f}")
                    
                    # Interpretation
                    st.markdown("### Interpretation")
                    if result['is_fake']:
                        if confidence_pct > 90:
                            st.warning("⚠️ **High Confidence Fake**: This image shows strong indicators of manipulation.")
                        elif confidence_pct > 70:
                            st.warning("⚠️ **Likely Fake**: Image shows signs of synthetic generation.")
                        else:
                            st.info("ℹ️ **Possibly Fake**: Borderline case, manual review recommended.")
                    else:
                        if confidence_pct > 90:
                            st.success("✅ **High Confidence Real**: Image appears authentic.")
                        elif confidence_pct > 70:
                            st.success("✅ **Likely Real**: Image shows natural characteristics.")
                        else:
                            st.info("ℹ️ **Possibly Real**: Borderline case, manual review recommended.")
                    
                    # Technical details (expandable)
                    with st.expander("🔧 Technical Details"):
                        st.markdown(f"""
                        **Prediction Pipeline:**
                        1. Face detection using Haar Cascade
                        2. Face cropping with 20% padding
                        3. Image resized to 256×256 pixels
                        4. Normalized to [-1, 1] range (built-in)
                        5. Processed through EfficientNetV2B0 backbone
                        6. Classification through 5-layer dense network
                        7. Sigmoid output: {result['raw_score']:.6f}
                        
                        **Threshold:** 0.5 (scores ≥0.5 = Fake, <0.5 = Real)
                        
                        **Model Capabilities:**
                        - Trained on 40,000 diverse face images
                        - Detects artifacts from multiple deepfake algorithms
                        - Robust to lighting, angle, and quality variations
                        - Automatic face detection and cropping for optimal analysis
                        """)
        else:
            st.info("👆 Upload an image to begin analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>⚡ Powered by TensorFlow & EfficientNetV2B0 | 🎯 96.5% Accuracy</p>
        <p style='font-size: 0.8rem;'>This model achieves 94.7% fake detection and 98.3% real detection rates.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
