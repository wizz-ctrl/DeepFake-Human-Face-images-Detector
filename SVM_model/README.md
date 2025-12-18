# AI-Generated Fake Face Detection System

This project implements a supervised learning model to detect AI-generated fake faces using feature engineering techniques.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Dataset Structure
Ensure your dataset is organized as follows:
```
ML Project/
├── dataset/
│   ├── 0/          # Real face images
│   └── 1/          # Fake (AI-generated) face images
├── train_model.py
├── predict.py
└── requirements.txt
```

## How to Use

### Step 1: Train the Model
Run the training script to train the fake face detection model:

```bash
python train_model.py
```

This will:
- Load images from the `dataset` folder
- Extract features using multiple techniques (color histograms, texture, edges, frequency domain)
- Train a Random Forest classifier
- Display evaluation metrics (accuracy, precision, recall, F1-score)
- Save the trained model as `fake_face_detector.pkl`
- Generate visualization plots (`confusion_matrix.png`, `feature_importance.png`)

### Step 2: Make Predictions

#### Predict a Single Image:
```bash
python predict.py path/to/your/image.jpg
```

#### Predict All Images in a Folder:
```bash
python predict.py --folder path/to/folder
```

## Features Extracted

The model uses the following feature engineering techniques:

1. **Color Histogram Features**: RGB channel distributions
2. **Texture Features**: Local Binary Patterns (LBP)
3. **Edge Features**: Edge density using Canny edge detection
4. **Frequency Domain Features**: Discrete Cosine Transform (DCT)
5. **Statistical Features**: Mean, std, median, percentiles
6. **Gradient Features**: Sobel gradient magnitudes

## Model Details

- **Algorithm**: Random Forest Classifier
- **Feature Vector Size**: ~400+ features per image
- **Preprocessing**: StandardScaler normalization
- **Train/Test Split**: 80/20 with stratification

## Output Files

After training, you'll get:
- `fake_face_detector.pkl` - Trained model
- `scaler.pkl` - Feature scaler
- `confusion_matrix.png` - Confusion matrix visualization
- `feature_importance.png` - Feature importance plot

## Tips for Best Results

1. Ensure balanced dataset (similar number of real and fake images)
2. Use high-quality images (at least 128x128 pixels)
3. Include diverse face images (different angles, lighting, backgrounds)
4. Minimum recommended: 100+ images per class

## Troubleshooting

- If accuracy is low, try collecting more training data
- Ensure images are clear and faces are visible
- Check that dataset folder structure is correct
