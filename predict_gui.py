import tensorflow as tf
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

class DeepfakeDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Deepfake Detector")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # Load model
        self.model = None
        self.load_model()
        
        # Setup GUI
        self.setup_gui()
    
    def load_model(self):
        """Load the trained model"""
        try:
            print("Loading model...")
            self.model = tf.keras.models.load_model('deepfake_detector_v4_improved.h5')
            print("✅ Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.root.destroy()
    
    def setup_gui(self):
        """Setup the GUI components"""
        # Title
        title_label = tk.Label(
            self.root,
            text="🔍 DEEPFAKE DETECTOR",
            font=("Arial", 24, "bold"),
            fg="#2c3e50"
        )
        title_label.pack(pady=20)
        
        # Subtitle
        subtitle = tk.Label(
            self.root,
            text="Detect AI-Generated vs Real Face Images",
            font=("Arial", 12),
            fg="#7f8c8d"
        )
        subtitle.pack()
        
        # Upload button
        self.upload_btn = tk.Button(
            self.root,
            text="SELECT IMAGE",
            command=self.select_image,
            font=("Arial", 14, "bold"),
            bg="#3498db",
            fg="white",
            padx=30,
            pady=15,
            cursor="hand2"
        )
        self.upload_btn.pack(pady=30)
        
        # Image preview frame
        self.preview_frame = tk.Frame(self.root, bg="#ecf0f1", width=300, height=200)
        self.preview_frame.pack(pady=10)
        self.preview_frame.pack_propagate(False)
        
        self.preview_label = tk.Label(self.preview_frame, text="No image selected", bg="#ecf0f1", font=("Arial", 10))
        self.preview_label.pack(expand=True)
        
        # Result frame
        self.result_frame = tk.Frame(self.root)
        self.result_frame.pack(pady=20)
        
        self.result_label = tk.Label(
            self.result_frame,
            text="",
            font=("Arial", 16, "bold")
        )
        self.result_label.pack()
        
        self.confidence_label = tk.Label(
            self.result_frame,
            text="",
            font=("Arial", 12)
        )
        self.confidence_label.pack()
        
        self.score_label = tk.Label(
            self.result_frame,
            text="",
            font=("Arial", 10),
            fg="#7f8c8d"
        )
        self.score_label.pack()
    
    def select_image(self):
        """Open file dialog to select image"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.predict_image(file_path)
    
    def predict_image(self, image_path):
        """Predict if image is Real or Fake"""
        try:
            # Load and display image preview
            img = Image.open(image_path).convert('RGB')
            img_display = img.copy()
            img_display.thumbnail((280, 180))
            
            # Update preview
            from PIL import ImageTk
            photo = ImageTk.PhotoImage(img_display)
            self.preview_label.config(image=photo, text="")
            self.preview_label.image = photo
            
            # Preprocess for model
            # Model has built-in Rescaling layer, so just resize and convert to array
            img_resized = img.resize((256, 256))
            img_array = np.array(img_resized, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            # DO NOT normalize here - model does it internally!
            
            # Predict
            prediction = self.model.predict(img_array, verbose=0)[0][0]
            
            # Display results
            # Model outputs: 0=Real (folder 0), 1=Fake (folder 1)
            # So LOW scores = Real, HIGH scores = Fake
            if prediction < 0.5:
                result_text = "✅ REAL"
                result_color = "#27ae60"
                confidence = (1 - prediction) * 100
            else:
                result_text = "🚨 FAKE (AI-Generated)"
                result_color = "#e74c3c"
                confidence = prediction * 100
            
            self.result_label.config(text=result_text, fg=result_color)
            self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")
            self.score_label.config(text=f"Raw Score: {prediction:.4f} (<0.5=Real, >=0.5=Fake)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {e}")
            self.result_label.config(text="")
            self.confidence_label.config(text="")
            self.score_label.config(text="")


if __name__ == "__main__":
    root = tk.Tk()
    app = DeepfakeDetectorGUI(root)
    root.mainloop()
