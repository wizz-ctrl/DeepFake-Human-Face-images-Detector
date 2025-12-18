import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models
import os

# Dataset directory
data_dir = "/home/wizz/ML Project/Dataset"
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validate')

# Parameters
img_size = (224, 224)
batch_size = 32
num_classes = 2  # You have 2 classes: 0 and 1

print("Loading training dataset...")
train_ds = image_dataset_from_directory(
    train_dir,
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

print("Loading validation dataset...")
val_ds = image_dataset_from_directory(
    val_dir,
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

# Get class names
class_names = train_ds.class_names
print(f"\nClass names: {class_names}")

# Configure auto-tuning for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Build the model
print("\nBuilding model...")
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),

    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train the model
print("\nStarting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)

# Save the model
model.save('trained_model.h5')
print("\nModel saved as 'trained_model.h5'")

# Print final results
print(f"\nFinal Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
