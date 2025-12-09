# Deploying Your Deepfake Detector to Hugging Face 🚀

## Step 1: Test Locally

```bash
cd "/home/wizz/ML Project"
streamlit run app.py
```

Open your browser to `http://localhost:8501` to test the app.

---

## Step 2: Create Hugging Face Account

1. Go to https://huggingface.co/join
2. Sign up for a free account
3. Verify your email

---

## Step 3: Install Git LFS (for large model files)

```bash
# Install Git LFS
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# Initialize Git LFS
git lfs install
```

---

## Step 4: Create New Space on Hugging Face

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Configure:
   - **Space name:** `deepfake-detector` (or your choice)
   - **License:** MIT
   - **SDK:** Streamlit
   - **Hardware:** CPU (free tier - sufficient for inference)
   - **Visibility:** Public or Private

---

## Step 5: Clone the Space Repository

```bash
# Clone your space (replace USERNAME with your HuggingFace username)
git clone https://huggingface.co/spaces/USERNAME/deepfake-detector
cd deepfake-detector

# Copy your files
cp "/home/wizz/ML Project/app.py" .
cp "/home/wizz/ML Project/requirements.txt" .
cp "/home/wizz/ML Project/README.md" .
cp "/home/wizz/ML Project/.gitattributes" .
cp "/home/wizz/ML Project/deepfake_detector_v4_40k.h5" .

# Optional: Include backup model
cp "/home/wizz/ML Project/best_recall_phase2.h5" .
```

---

## Step 6: Track Large Files with Git LFS

```bash
# Track .h5 model files with Git LFS
git lfs track "*.h5"

# Add .gitattributes
git add .gitattributes
```

---

## Step 7: Commit and Push to Hugging Face

```bash
# Add all files
git add app.py requirements.txt README.md
git add deepfake_detector_v4_40k.h5

# Commit
git commit -m "Initial commit: Deepfake Detector with 96.5% accuracy"

# Push to Hugging Face
git push origin main
```

This will take a few minutes because the model file is ~94MB.

---

## Step 8: Wait for Build

1. Go to your space URL: `https://huggingface.co/spaces/USERNAME/deepfake-detector`
2. Wait for the build to complete (5-10 minutes)
3. Once built, your app will be live!

---

## Step 9: Share Your Space

Your app will be available at:
```
https://huggingface.co/spaces/USERNAME/deepfake-detector
```

Share this URL with anyone!

---

## Troubleshooting

### Issue: Model file too large (>100MB)

If your model exceeds GitHub's limits:

```bash
# Install Git LFS first
git lfs install

# Track the model file
git lfs track "deepfake_detector_v4_40k.h5"
git add .gitattributes
git add deepfake_detector_v4_40k.h5
git commit -m "Add model with Git LFS"
git push
```

### Issue: Out of memory during inference

Update `app.py` to use smaller batch size or add caching:

```python
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('deepfake_detector_v4_40k.h5')
    return model
```

### Issue: Slow startup time

Add a loading message in `app.py`:

```python
with st.spinner("Loading AI model... (first load may take 30 seconds)"):
    model = load_model()
```

### Issue: TensorFlow version conflicts

If you get TensorFlow errors, update `requirements.txt`:

```txt
tensorflow==2.15.0
tensorflow-cpu==2.15.0  # Use CPU version for faster cold starts
```

---

## Advanced: Custom Domain (Optional)

Hugging Face Spaces provides a subdomain, but you can:
1. Use Cloudflare or Netlify for custom domain
2. Point CNAME to your space URL
3. Set up reverse proxy

---

## Updating Your Deployed App

To update the app after deployment:

```bash
cd deepfake-detector

# Make changes to app.py locally
nano app.py

# Commit and push
git add app.py
git commit -m "Update UI/fix bugs"
git push origin main
```

Hugging Face will automatically rebuild your space.

---

## Monitoring Usage

1. Go to your space settings
2. Check "Analytics" tab to see:
   - Number of visitors
   - API calls
   - Resource usage

---

## Cost Considerations

- **Free tier:** CPU inference (sufficient for your model)
- **Upgrade options:** 
  - GPU ($0.60/hour) - unnecessary for your model
  - More CPU/RAM if needed
  - Private spaces for team use

Your current model runs fine on free CPU tier! 🎉

---

## Next Steps

1. **Test locally** with `streamlit run app.py`
2. **Create HuggingFace account** and space
3. **Push your files** following steps above
4. **Share your deployed app** URL!

Need help? Check:
- Streamlit docs: https://docs.streamlit.io
- Hugging Face Spaces: https://huggingface.co/docs/hub/spaces
