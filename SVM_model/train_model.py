import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import ndimage
from scipy.fftpack import dct, fft2
from multiprocessing import Pool, cpu_count, freeze_support
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("Warning: XGBoost not installed. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False
try:
    from skimage.feature import hog
    SKIMAGE_AVAILABLE = True
except ImportError:
    print("Warning: scikit-image not installed. Install with: pip install scikit-image")
    SKIMAGE_AVAILABLE = False

def augment_image(img):
    """Apply minimal augmentation to preserve image characteristics"""
    augmented = []
    
    # Original (most important)
    augmented.append(img.copy())
    
    # Horizontal flip (safe for faces)
    augmented.append(cv2.flip(img, 1))
    
    # Very slight brightness adjustment only
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.05, 0, 255)  # Slight brighten
        hsv = hsv.astype(np.uint8)
        augmented.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 0.95, 0, 255)  # Slight darken
        hsv = hsv.astype(np.uint8)
        augmented.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    except:
        pass
    
    return augmented

def extract_features(image_path, augment=False):
    """Extract multiple features from an image for fake face detection"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Balanced image size - 160x160 is optimal for fake detection
    img = cv2.resize(img, (160, 160))
    
    if augment:
        # Apply augmentation only during training
        images = augment_image(img)
        all_features = []
        for aug_img in images:
            features = extract_single_image_features(aug_img)
            if features is not None:
                all_features.append(features)
        return all_features if all_features else None
    else:
        return extract_single_image_features(img)

def extract_single_image_features(img):
    """Extract features from a single image"""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features = []
        
        # 1. Color histogram features (more bins for better discrimination)
        for i in range(3):
            hist = cv2.calcHist([img], [i], None, [32], [0, 256])
            features.extend(hist.flatten())
        
        # 2. LBP features with multiple radii for better texture analysis
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
                # For radius=2, we need to slice properly to match center size
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
        
        # 3. Multiple edge detection methods
        edges_canny = cv2.Canny(gray, 100, 200)
        edge_density_canny = np.sum(edges_canny) / (edges_canny.shape[0] * edges_canny.shape[1])
        features.append(edge_density_canny)
        
        # Laplacian edges
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.extend([np.mean(np.abs(laplacian)), np.std(laplacian)])
        
        # 4. Larger DCT features for frequency analysis
        dct_img = dct(dct(gray.T, norm='ortho').T, norm='ortho')
        dct_features = dct_img[:24, :24].flatten()  # More DCT coefficients for larger image
        features.extend(dct_features)
        
        # 5. Enhanced statistical features
        features.extend([
            np.mean(gray), np.std(gray), np.median(gray),
            np.percentile(gray, 25), np.percentile(gray, 75),
            np.min(gray), np.max(gray),
            np.percentile(gray, 10), np.percentile(gray, 90)
        ])
        
        # 6. Gradient features in multiple directions
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        gradient_direction = np.arctan2(gy, gx)
        
        features.extend([
            np.mean(gradient_magnitude), np.std(gradient_magnitude),
            np.max(gradient_magnitude), np.median(gradient_magnitude)
        ])
        
        # Gradient direction histogram
        dir_hist, _ = np.histogram(gradient_direction, bins=16, range=(-np.pi, np.pi))
        features.extend(dir_hist)
        
        # 7. Color channel statistics
        for channel in cv2.split(img):
            features.extend([np.mean(channel), np.std(channel)])
        
        # 8. GLCM-inspired features (simplified co-occurrence)
        h_diff = np.abs(gray[:, 1:].astype(np.int32) - gray[:, :-1].astype(np.int32))
        features.extend([np.mean(h_diff), np.std(h_diff)])
        
        v_diff = np.abs(gray[1:, :].astype(np.int32) - gray[:-1, :].astype(np.int32))
        features.extend([np.mean(v_diff), np.std(v_diff)])
        
        # 9. HOG features (if available) - more features with larger image
        if SKIMAGE_AVAILABLE:
            hog_features = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                              cells_per_block=(2, 2), visualize=False)
            features.extend(hog_features[:200])  # More HOG features for larger image
        
        # 10. FFT features for frequency analysis
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
        
        # 11. HSV color space features
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for channel in range(3):
            hist = cv2.calcHist([hsv], [channel], None, [16], [0, 256])
            features.extend(hist.flatten())
        
        # 12. LAB color space features
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        for channel in range(3):
            features.extend([np.mean(lab[:,:,channel]), np.std(lab[:,:,channel])])
        
        # 13. Noise estimation (high-frequency content)
        high_pass = gray.astype(np.float32) - cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
        features.extend([np.mean(np.abs(high_pass)), np.std(high_pass)])
        
        # 14. Blur detection (variance of Laplacian)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features.append(laplacian_var)
        
        return np.array(features)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def process_image_wrapper(args):
    """Wrapper for multiprocessing"""
    img_path, label, augment = args
    features = extract_features(img_path, augment=augment)
    if features is not None:
        if augment and isinstance(features, list):
            # Return multiple augmented versions
            return [(f, label) for f in features]
        return [(features, label)]
    return None

def load_dataset(dataset_path, subset='train', use_multiprocessing=True, augment=False):
    """Load images and labels from dataset folder with parallel processing"""
    
    # Collect all image paths
    real_path = os.path.join(dataset_path, subset, '0')
    fake_path = os.path.join(dataset_path, subset, '1')
    
    print(f"Loading images from {real_path} and {fake_path}...")
    
    if not os.path.exists(real_path) or not os.path.exists(fake_path):
        print(f"Error: Paths do not exist!")
        return np.array([]), np.array([])
    
    # Gather all image paths with labels
    image_tasks = []
    for img_name in os.listdir(real_path):
        img_path = os.path.join(real_path, img_name)
        if os.path.isfile(img_path):
            image_tasks.append((img_path, 0, augment))
    
    for img_name in os.listdir(fake_path):
        img_path = os.path.join(fake_path, img_name)
        if os.path.isfile(img_path):
            image_tasks.append((img_path, 1, augment))
    
    print(f"Total images to process: {len(image_tasks)}")
    if augment:
        print("Data augmentation enabled (4x samples)")
    
    X = []
    y = []
    
    if use_multiprocessing:
        num_workers = max(1, cpu_count() - 1)  # Use all CPU cores except one
        print(f"Using {num_workers} CPU cores for parallel processing...")
        
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_image_wrapper, image_tasks),
                total=len(image_tasks),
                desc=f"Processing {subset} images"
            ))
        
        # Filter out None results
        for result in results:
            if result is not None:
                for features, label in result:
                    X.append(features)
                    y.append(label)
    else:
        # Sequential processing with progress bar
        for img_path, label, aug in tqdm(image_tasks, desc=f"Processing {subset} images"):
            features = extract_features(img_path, augment=aug)
            if features is not None:
                if aug and isinstance(features, list):
                    for f in features:
                        X.append(f)
                        y.append(label)
                else:
                    X.append(features)
                    y.append(label)
    
    return np.array(X), np.array(y)

def train_and_evaluate():
    """Main training pipeline with proper train/validate/test split"""
    dataset_path = 'Dataset'
    
    # Load all three datasets
    print("="*60)
    print("LOADING TRAINING DATASET (with augmentation)")
    print("="*60)
    X_train, y_train = load_dataset(dataset_path, 'train', use_multiprocessing=True, augment=True)
    
    print("\n" + "="*60)
    print("LOADING VALIDATION DATASET (no augmentation)")
    print("="*60)
    X_val, y_val = load_dataset(dataset_path, 'validate', use_multiprocessing=True, augment=False)
    
    print("\n" + "="*60)
    print("LOADING TEST DATASET (no augmentation)")
    print("="*60)
    X_test, y_test = load_dataset(dataset_path, 'test', use_multiprocessing=True, augment=False)
    
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Training set: {len(X_train)} images (augmented)")
    print(f"  Real: {np.sum(y_train == 0)}, Fake: {np.sum(y_train == 1)}")
    
    print(f"\nValidation set: {len(X_val)} images")
    print(f"  Real: {np.sum(y_val == 0)}, Fake: {np.sum(y_val == 1)}")
    
    print(f"\nTest set: {len(X_test)} images")
    print(f"  Real: {np.sum(y_test == 0)}, Fake: {np.sum(y_test == 1)}")
    
    print(f"\nFeature vector size: {X_train.shape[1]}")
    
    # Normalize features
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature Selection to reduce overfitting
    print("\n" + "="*60)
    print("FEATURE SELECTION (Optimized)")
    print("="*60)
    
    # Use Extra Trees for feature importance-based selection
    print("Selecting most important features...")
    feature_selector = SelectFromModel(
        ExtraTreesClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        threshold='0.3*mean'  # Keep more features for better accuracy
    )
    feature_selector.fit(X_train_scaled, y_train)
    
    X_train_selected = feature_selector.transform(X_train_scaled)
    X_val_selected = feature_selector.transform(X_val_scaled)
    X_test_selected = feature_selector.transform(X_test_scaled)
    
    print(f"Features reduced: {X_train_scaled.shape[1]} -> {X_train_selected.shape[1]}")
    print(f"Removed {X_train_scaled.shape[1] - X_train_selected.shape[1]} low-importance features")
    
    # Train multiple models with MUCH stronger regularization
    print("\n" + "="*60)
    print("TRAINING MODELS (Optimized for ~80% accuracy)")
    print("="*60)
    
    # Random Forest - optimized
    rf_model = RandomForestClassifier(
        n_estimators=500,          # More trees for stability
        max_depth=18,              # Slightly deeper
        min_samples_split=8,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced',   # Handle class imbalance
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Gradient Boosting - optimized
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.08,        # Slightly higher LR
        subsample=0.85,
        min_samples_split=8,
        min_samples_leaf=4,
        max_features=0.7,          # More features
        random_state=42,
        verbose=1
    )
    
    # Extra Trees - optimized
    et_model = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=18,
        min_samples_split=8,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # SVM with RBF kernel - good for this task
    svm_model = SVC(
        C=10,
        kernel='rbf',
        gamma='scale',
        probability=True,          # Needed for soft voting
        class_weight='balanced',
        random_state=42
    )
    
    models = {
        'Random Forest': rf_model,
        'Gradient Boosting': gb_model,
        'Extra Trees': et_model,
        'SVM': svm_model
    }
    
    best_model = None
    best_accuracy = 0
    best_model_name = ""
    trained_models = {}
    
    # Cross-validation for more robust evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        print(f"{model_name} CV Score: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
        
        # Fit on full training data
        model.fit(X_train_selected, y_train)
        trained_models[model_name] = model
        
        # Validate
        y_val_pred = model.predict(X_val_selected)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        # Check for overfitting
        train_accuracy = accuracy_score(y_train, model.predict(X_train_selected))
        overfitting_gap = train_accuracy - val_accuracy
        
        print(f"{model_name} Results:")
        print(f"  Train Accuracy: {train_accuracy*100:.2f}%")
        print(f"  Val Accuracy: {val_accuracy*100:.2f}%")
        print(f"  Overfitting Gap: {overfitting_gap*100:.2f}%")
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model
            best_model_name = model_name
    
    # Create and properly train ensemble (voting classifier)
    print(f"\nCreating and training ensemble model...")
    ensemble_estimators = [
        ('rf', RandomForestClassifier(
            n_estimators=500,
            max_depth=18,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features=0.7,
            random_state=42
        )),
        ('et', ExtraTreesClassifier(
            n_estimators=500,
            max_depth=18,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )),
        ('svm', SVC(
            C=10,
            kernel='rbf',
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        ))
    ]
    
    ensemble = VotingClassifier(
        estimators=ensemble_estimators,
        voting='soft',
        n_jobs=-1
    )
    
    # Properly fit the ensemble
    print("Fitting ensemble...")
    ensemble.fit(X_train_selected, y_train)
    
    # Evaluate ensemble
    y_val_ensemble = ensemble.predict(X_val_selected)
    ensemble_accuracy = accuracy_score(y_val, y_val_ensemble)
    ensemble_train_accuracy = accuracy_score(y_train, ensemble.predict(X_train_selected))
    
    print(f"Ensemble Results:")
    print(f"  Train Accuracy: {ensemble_train_accuracy*100:.2f}%")
    print(f"  Val Accuracy: {ensemble_accuracy*100:.2f}%")
    print(f"  Overfitting Gap: {(ensemble_train_accuracy - ensemble_accuracy)*100:.2f}%")
    
    # Choose best between individual models and ensemble
    if ensemble_accuracy > best_accuracy:
        best_model = ensemble
        best_model_name = "Ensemble (RF + GB + ET + SVM)"
        best_accuracy = ensemble_accuracy
    
    print(f"\n{'='*60}")
    print(f"Best model: {best_model_name}")
    print(f"Validation Accuracy: {best_accuracy*100:.2f}%")
    print("="*60)
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL TEST SET EVALUATION")
    print("="*60)
    y_pred = best_model.predict(X_test_selected)
    
    test_accuracy = accuracy_score(y_test, y_pred)
    train_accuracy_final = accuracy_score(y_train, best_model.predict(X_train_selected))
    
    print(f"\nFinal Results:")
    print(f"  Train Accuracy: {train_accuracy_final*100:.2f}%")
    print(f"  Validation Accuracy: {best_accuracy*100:.2f}%")
    print(f"  Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"  Generalization Gap: {(train_accuracy_final - test_accuracy)*100:.2f}%")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title(f'Confusion Matrix - {best_model_name}\nTest Accuracy: {test_accuracy*100:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    
    # Feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = best_model.feature_importances_
        top_20_indices = np.argsort(feature_importance)[-20:]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(20), feature_importance[top_20_indices])
        plt.xlabel('Importance Score')
        plt.ylabel('Feature Index')
        plt.title(f'Top 20 Most Important Features - {best_model_name}')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Feature importance plot saved as 'feature_importance.png'")
    
    # Save best model, scaler, and feature selector
    joblib.dump(best_model, 'fake_face_detector.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(feature_selector, 'feature_selector.pkl')
    
    metadata = {
        'model_type': best_model_name,
        'train_accuracy': train_accuracy_final,
        'val_accuracy': best_accuracy,
        'test_accuracy': test_accuracy,
        'original_features': X_train_scaled.shape[1],
        'selected_features': X_train_selected.shape[1],
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'overfitting_gap': train_accuracy_final - test_accuracy
    }
    joblib.dump(metadata, 'model_metadata.pkl')
    
    print("\n" + "="*60)
    print("FILES SAVED")
    print("="*60)
    print("✓ fake_face_detector.pkl - Trained model")
    print("✓ scaler.pkl - Feature scaler")
    print("✓ model_metadata.pkl - Model information")
    print("✓ confusion_matrix.png - Performance visualization")
    print("✓ feature_importance.png - Feature analysis")
    print("="*60)
    
    return best_model, scaler

if __name__ == "__main__":
    freeze_support()  # Required for Windows multiprocessing
    train_and_evaluate()
