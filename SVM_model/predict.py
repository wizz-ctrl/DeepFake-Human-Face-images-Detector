import os
import numpy as np
import cv2
import joblib
from train_model import extract_features

def predict_image(image_path, model, scaler, feature_selector=None):
    """Predict if an image is real or fake"""
    features = extract_features(image_path)
    
    if features is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    # Reshape and scale
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Apply feature selection if available
    if feature_selector is not None:
        features_scaled = feature_selector.transform(features_scaled)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return prediction, probability

def predict_single_image(image_path):
    """Predict a single image"""
    # Load model, scaler, and feature selector
    try:
        model = joblib.load('fake_face_detector.pkl')
        scaler = joblib.load('scaler.pkl')
        try:
            feature_selector = joblib.load('feature_selector.pkl')
        except FileNotFoundError:
            feature_selector = None
    except FileNotFoundError:
        print("Error: Model files not found. Please train the model first by running train_model.py")
        return
    
    prediction, probability = predict_image(image_path, model, scaler, feature_selector)
    
    if prediction is not None:
        label = "FAKE (AI-Generated)" if prediction == 1 else "REAL"
        confidence = probability[prediction] * 100
        
        print(f"\nImage: {image_path}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Probability [Real: {probability[0]:.4f}, Fake: {probability[1]:.4f}]")
        
        # Display image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (400, 400))
        cv2.putText(img, f"{label} ({confidence:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if prediction == 0 else (0, 0, 255), 2)
        cv2.imshow('Prediction', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def predict_folder(folder_path):
    """Predict all images in a folder"""
    try:
        model = joblib.load('fake_face_detector.pkl')
        scaler = joblib.load('scaler.pkl')
        try:
            feature_selector = joblib.load('feature_selector.pkl')
        except FileNotFoundError:
            feature_selector = None
    except FileNotFoundError:
        print("Error: Model files not found. Please train the model first.")
        return
    
    results = []
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            prediction, probability = predict_image(img_path, model, scaler, feature_selector)
            if prediction is not None:
                results.append({
                    'image': img_name,
                    'prediction': 'FAKE' if prediction == 1 else 'REAL',
                    'confidence': probability[prediction] * 100
                })
    
    print("\nPrediction Results:")
    print("-" * 60)
    for result in results:
        print(f"{result['image']}: {result['prediction']} ({result['confidence']:.2f}%)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python predict.py <image_path>          # Predict single image")
        print("  python predict.py --folder <folder_path> # Predict all images in folder")
    elif sys.argv[1] == '--folder' and len(sys.argv) == 3:
        predict_folder(sys.argv[2])
    else:
        predict_single_image(sys.argv[1])
