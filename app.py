# app.py
import os
import io
import tempfile
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# 3rd-party helper to download from Google Drive robustly
# We'll try to import gdown; if not present, fallback to a requests-based downloader.
try:
    import gdown
    HAS_GDOWN = True
except Exception:
    HAS_GDOWN = False
import requests
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# -----------------------
# User-configurable
# -----------------------
# Set input image size used in training (must match training)
INPUT_SIZE = (240, 240)   # if you trained on 240x240 set this to (240,240)
INPUT_SHAPE = INPUT_SIZE + (3,)
HEAD_UNITS = 128          # must match the head you used when training
NUM_CLASSES = 5        # optional: set to an integer here if known

# Google Drive file id for weights
# ORIGINAL URL:
# https://drive.google.com/file/d/1of6uxoLlYOSvtc4kP0wUlcvtI-mWsKt_/view?usp=sharing
# File id is the part after '/d/' and before '/view' -> "1of6uxoLlYOSvtc4kP0wUlcvtI-mWsKt_"
GDRIVE_FILE_ID = "1of6uxoLlYOSvtc4kP0wUlcvtI-mWsKt_"
WEIGHTS_FILE_NAME = "final_weights_finetuned.weights.h5"

MODEL_DIR = "model_data"
os.makedirs(MODEL_DIR, exist_ok=True)
WEIGHTS_LOCAL_PATH = os.path.join(MODEL_DIR, WEIGHTS_FILE_NAME)

# -----------------------
# Utility: download from Google Drive
# -----------------------
def download_from_gdrive(file_id: str, dest: str, use_gdown=HAS_GDOWN):
    """
    Downloads a file from Google Drive. Uses gdown if available (recommended).
    If gdown is missing, uses a requests-based method that handles confirmation tokens.
    """
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest  # already downloaded

    if use_gdown:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest, quiet=False, fuzzy=True)
        return dest

    # fallback method (handles large files with confirm token)
    session = requests.Session()
    url = "https://docs.google.com/uc?export=download"
    response = session.get(url, params={'id': file_id}, stream=True)
    token = _get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)

    _save_response_content(response, dest)
    return dest

def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    # fallback: check for confirm token in page (less reliable)
    return None

def _save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)

# -----------------------
# Build the model architecture (must match training)
# -----------------------
@st.cache_resource(show_spinner=False)
def build_model(input_shape=INPUT_SHAPE, head_units=HEAD_UNITS, num_classes=None):
    """Build MobileNetV2 base + GAP + small dense head. num_classes can be None; we'll infer later if possible."""
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(head_units, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    # Create a placeholder output with some classes; we will rebuild final layer when we know class count
    if num_classes is None:
        # temporary small output; will be replaced when weights loaded if needed
        outputs = layers.Dense(1, activation='linear')(x)
        model = models.Model(inputs=base.input, outputs=outputs)
        return model, base
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=base.input, outputs=outputs)
        return model, base

# -----------------------
# Load model and weights (cached)
# -----------------------
@st.cache_resource(show_spinner=True)
def load_model_and_weights(gdrive_file_id, weights_local_path, input_shape=INPUT_SHAPE, head_units=HEAD_UNITS, num_classes=NUM_CLASSES):
    # 1) Ensure weights file downloaded
    try:
        download_from_gdrive(gdrive_file_id, weights_local_path)
    except Exception as e:
        st.error(f"Failed to download weights from Google Drive: {e}")
        raise

    # 2) Build model skeleton
    model_skel, base = build_model(input_shape=input_shape, head_units=head_units, num_classes=num_classes)

    # If num_classes was None in build, we create a model with the correct final layer based on the weights' shape:
    # We'll attempt to load weights into the model carefully. The saved weights are weights-only, so the model
    # architecture must match exactly. If the saved head uses NUM_CLASSES=K and our model_skel used 1 output,
    # we need to rebuild the final layer accordingly.
    # A safe approach: try loading weights directly. If it fails because of mismatch, attempt to infer number of classes
    # by inspecting the weights HDF5 file and rebuild the head.

    # Try direct load first:
    try:
        model_skel.load_weights(weights_local_path)
        st.info("Loaded weights into model successfully.")
        return model_skel
    except Exception as e:
        # attempt to inspect weights h5 to discover final Dense layer shape
        try:
            import h5py
            with h5py.File(weights_local_path, 'r') as hf:
                # weights saved by model.save_weights have groups like:
                # 'dense/...', 'batch_normalization/...', etc.
                # We'll search for the final Dense layer weight dataset shape.
                # This is heuristic but often works: find the largest dense weight matrix among top-level groups.
                candidate = None
                for key in hf.keys():
                    # dive one level in; groups usually named like 'dense' or 'dense_1'
                    if isinstance(hf[key], h5py.Group):
                        # inspect subgroups datasets
                        for subk in hf[key].keys():
                            ds = hf[key][subk]
                            if isinstance(ds, h5py.Dataset):
                                shape = ds.shape
                                # weights matrix (kernel) usually 2-D
                                if len(shape) == 2:
                                    # candidate for output units is shape[1] if last dense stored as kernel shape (in Keras it's (in_dim, out_dim))
                                    candidate = shape[1]
                inferred_num_classes = int(candidate) if candidate is not None else None
        except Exception:
            inferred_num_classes = None

        if inferred_num_classes is None:
            st.error("Failed to load weights and could not infer final layer size from the weights file. "
                     "Please ensure the model architecture in the app exactly matches the one used during training "
                     "and that the weights file is the one produced by model.save_weights(...).")
            raise e

        # Rebuild model with inferred final classes and load weights
        model_full, _ = build_model(input_shape=input_shape, head_units=head_units, num_classes=inferred_num_classes)
        model_full.load_weights(weights_local_path)
        st.info(f"Rebuilt model with inferred num_classes={inferred_num_classes} and loaded weights.")
        return model_full

# -----------------------
# Preprocess and predict
# -----------------------
def preprocess_image(image: Image.Image, target_size=INPUT_SIZE):
    image = image.convert("RGB")
    image = image.resize(target_size)
    arr = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# -----------------------
# Streamlit layout
# -----------------------
st.set_page_config(page_title="Sugarcane Age Classifier", layout="centered")
st.title("Sugarcane Age Classifier (MobileNetV2)")

st.sidebar.header("Model / Weights")
st.sidebar.write("Weights source: Google Drive (saved with `model.save_weights(...)`)")

with st.sidebar.expander("Advanced"):
    st.write("If you have a local weights file, you can upload it to replace the Drive download.")
    uploaded_wts = st.file_uploader("Upload local weights (.h5)", type=["h5", "hdf5", "weights"])
    if uploaded_wts is not None:
        # save to local path and override
        tmpw = os.path.join(MODEL_DIR, "uploaded_weights.h5")
        with open(tmpw, "wb") as f:
            f.write(uploaded_wts.getbuffer())
        WEIGHTS_LOCAL_OVERRIDE = tmpw
    else:
        WEIGHTS_LOCAL_OVERRIDE = None

st.sidebar.markdown("---")
st.sidebar.write("If classifier labels are known, paste them (comma-separated) below; "
                 "otherwise predictions will be shown as class indices.")
label_map_text = st.sidebar.text_area("11 Month, 2 Month, 4 Month, 6 Month, 9 Month, value="", height=80)
label_list = [s.strip() for s in label_map_text.split(",") if s.strip()]
if label_list:
    NUM_CLASSES = len(label_list)

# Load model + weights (cached)
weights_path_to_use = WEIGHTS_LOCAL_OVERRIDE if WEIGHTS_LOCAL_OVERRIDE else WEIGHTS_LOCAL_PATH

try:
    model = load_model_and_weights(GDRIVE_FILE_ID, weights_path_to_use, input_shape=INPUT_SHAPE, head_units=HEAD_UNITS, num_classes=NUM_CLASSES)
except Exception as e:
    st.error("Model load failed. See messages above.")
    st.stop()

st.write("Model loaded. Input size:", INPUT_SIZE)

# Upload image or use example
st.write("Upload an image of sugarcane (single plant/canopy).")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded image", use_column_width=True)
    input_arr = preprocess_image(img, target_size=INPUT_SIZE)
    preds = model.predict(input_arr)
    # if final layer is linear/regression placeholder, attempt to softmax it only if multi-dim
    if preds.ndim == 1 or (preds.ndim == 2 and preds.shape[1] == 1):
        st.warning("Model output appears to be a single value; verify that the app's model architecture matches your training model.")
        st.write("Raw output:", preds.tolist())
    else:
        probs = preds[0]
        # if model uses linear output but not softmax (unlikely), apply softmax
        if not np.allclose(probs.sum(), 1.0, atol=1e-3):
            probs = softmax(probs)
        # show top 5
        top_idx = probs.argsort()[::-1]
        st.subheader("Predictions")
        for i in top_idx[:min(5, len(probs))]:
            label = label_list[i] if label_list else f"Class {i}"
            st.write(f"{label}: {probs[i]*100:.2f}%")
else:
    st.info("Upload an image to get predictions. You can optionally paste class labels in the sidebar for readable outputs.")
