# streamlit_app.py
import os
import logging
import warnings
from pathlib import Path
from io import BytesIO, StringIO
from PIL import Image
import numpy as np
import zipfile
import tempfile

# Suppress noisy logs/warnings before importing TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # INFO=1, WARNING=2, ERROR=3 -> hide INFO/WARNING
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import streamlit as st
from scipy.special import softmax

# Import TensorFlow / Keras after suppression flags
import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown  # to download shared file from Google Drive

# Streamlit config & deprecation option
st.set_page_config(page_title="Web Application for Sugarcane Age Detection using Drone Imagery", layout="wide")
st.set_option('deprecation.showfileUploaderEncoding', False)

# -------- Configuration --------
DRIVE_FILE_ID_DEFAULT = "10JYTIb9CWNhGbhnBNEA1Yj8SVVqx5BjE"
DEFAULT_MODEL_FILENAME = "/tmp/model.keras"  # cached location
VALID_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
USE_VGG_PREPROCESS = False  # default; UI lets user change
TOP_K_DEFAULT = 3

# -------- Utilities (adapted & improved) --------
def get_model_input_size(model):
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list):
        shape = shape[0]
    if not shape:
        return (240, 240, 3)
    if len(shape) == 4:
        _, h, w, c = shape
        h = int(h) if (h is not None) else 240
        w = int(w) if (w is not None) else 240
        c = int(c) if (c is not None) else 3
        return (h, w, c)
    return (240, 240, 3)

def infer_num_classes_from_model(model):
    try:
        out_shape = model.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        if isinstance(out_shape, tuple) and len(out_shape) >= 2:
            n = out_shape[-1]
            if isinstance(n, int):
                return n
    except Exception:
        pass
    try:
        for layer in reversed(model.layers):
            if hasattr(layer, "units"):
                n = getattr(layer, "units")
                if isinstance(n, int) and n > 1:
                    return n
    except Exception:
        pass
    return None

def build_default_class_map(model, prefix="class_"):
    n = infer_num_classes_from_model(model)
    if n is None:
        return {}
    else:
        return {i: f"{prefix}{i}" for i in range(n)}

def preprocess_image_for_model_bytes(img_bytes, model, use_vgg=USE_VGG_PREPROCESS):
    expected_h, expected_w, expected_c = get_model_input_size(model)
    pil = Image.open(BytesIO(img_bytes)).convert("RGB")
    pil_resized = pil.resize((expected_w, expected_h), Image.BILINEAR)
    x = np.array(pil_resized).astype("float32")
    if use_vgg:
        from tensorflow.keras.applications.vgg16 import preprocess_input
        x = preprocess_input(x)
    else:
        x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x, pil_resized, (expected_h, expected_w, expected_c)

def predict_from_bytes(model, img_bytes, class_map=None, top_k=3, use_vgg=USE_VGG_PREPROCESS):
    if class_map is None:
        class_map = build_default_class_map(model)
    x, pil_img, used_size = preprocess_image_for_model_bytes(img_bytes, model, use_vgg=use_vgg)
    preds = model.predict(x, verbose=0)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    preds = preds[0] if (hasattr(preds, "ndim") and preds.ndim == 2 and preds.shape[0] == 1) else preds
    try:
        if preds.sum() > 1.0001 or preds.min() < 0:
            probs = softmax(preds)
        else:
            probs = preds
    except Exception:
        probs = preds
    top_idx = probs.argsort()[-top_k:][::-1]
    results = [(int(idx), class_map.get(int(idx), f"class_{idx}") if class_map else f"class_{idx}", float(probs[idx]))
               for idx in top_idx]
    return results, pil_img

def is_image_filename(fname: str):
    return fname.lower().endswith(VALID_IMG_EXTS)

# -------- Model loading helpers --------
def download_from_gdrive(file_id: str, dest_path: str, force=False):
    dest = Path(dest_path)
    if dest.exists() and not force:
        return str(dest)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(dest), quiet=False)
    return str(dest)

def load_model_preferred(path, convert_h5_to_keras=True, compile=False, custom_objects=None):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    ext = p.suffix.lower()
    if ext == ".keras":
        return load_model(str(p), compile=compile, custom_objects=custom_objects)
    elif ext == ".h5" and convert_h5_to_keras:
        model = load_model(str(p), compile=compile, custom_objects=custom_objects)
        keras_path = str(p.with_suffix(".keras"))
        model.save(keras_path)
        return load_model(keras_path, compile=compile, custom_objects=custom_objects)
    else:
        return load_model(str(p), compile=compile, custom_objects=custom_objects)

@st.cache_resource(show_spinner=False)
def get_model_from_drive_or_path(drive_file_id: str = DRIVE_FILE_ID_DEFAULT, local_path: str = DEFAULT_MODEL_FILENAME, force_download=False):
    dest = download_from_gdrive(drive_file_id, local_path, force=force_download)
    model = load_model_preferred(dest, convert_h5_to_keras=True, compile=False, custom_objects=None)
    return model, dest

# -------- Streamlit UI --------
st.title("Web Application for Sugarcane Age Detection using Drone Imagery")
st.markdown("**Developed by:** SVERI's College of Engineering, Pandharpur  \n**Research funding support from:** Rajiv Gandhi Science and Technology Commission, Government of Maharashtra")
st.markdown("---")

# Brief model summary paragraph requested by user
st.markdown(
    """
**About the Model (brief):**

This application uses a MobileNet backbone (MobileNetV2-style) fine-tuned on drone imagery of sugarcane fields.
The model was trained to classify sugarcane crop age into discrete classes (for example: 2_month, 4_month, 6_month, 9_month, 11_month).
The trained MobileNet features are followed by a custom classification head (dense layer(s) and a final softmax classification layer).
The app downloads the inference `.keras` model from a provided Google Drive link, runs predictions on uploaded images (or ZIPs of images),
and returns top-K predicted age classes with probabilities.
"""
)

with st.sidebar:
    st.header("Model / Prediction Settings")
    drive_id = st.text_input("Google Drive file id (share link id)", value=DRIVE_FILE_ID_DEFAULT)
    model_dest = st.text_input("Local model path", value=DEFAULT_MODEL_FILENAME)
    force_dl = st.checkbox("Force re-download model", value=False)
    top_k = st.number_input("Top K predictions", min_value=1, max_value=10, value=TOP_K_DEFAULT)
    use_vgg = st.checkbox("Use VGG preprocessing instead of /255.0", value=USE_VGG_PREPROCESS)

    st.markdown("---")
    st.markdown("**Class map (optional)** — JSON-like mapping `index: label` (example):")
    st.code('{\n  0: "11_month",\n  1: "2_month",\n  2: "4_month",\n  3: "6_month",\n  4: "9_month"\n}')
    custom_map_text = st.text_area("Paste mapping (leave empty to auto)", height=100)
    if custom_map_text.strip():
        try:
            class_map = eval(custom_map_text.strip(), {})
            if not isinstance(class_map, dict):
                st.warning("Mapping is not a dict. Falling back to auto map.")
                class_map = None
        except Exception as e:
            st.warning(f"Could not parse mapping: {e}. Falling back to auto map.")
            class_map = None
    else:
        class_map = None

# Load model
with st.spinner("Downloading and loading model (if needed)..."):
    try:
        model, model_path = get_model_from_drive_or_path(drive_file_id=drive_id, local_path=model_dest, force_download=force_dl)
    except Exception as e:
        st.error(f"Failed to download or load model: {e}")
        st.stop()

st.success(f"Model loaded from: {model_path}")

# Show readable model summary inside an expander (capture from model.summary())
with st.expander("Model summary (click to expand)"):
    s = StringIO()
    try:
        model.summary(print_fn=lambda x: s.write(x + "\n"))
        st.text(s.getvalue())
    except Exception as e:
        st.text(f"Could not display model summary: {e}")

# Files input
st.header("Upload files")
st.write("Upload single/multiple images or a single ZIP file containing images.")
uploaded = st.file_uploader("Choose images or a ZIP", accept_multiple_files=True, type=["jpg","jpeg","png","bmp","tiff","zip"])

if uploaded:
    results_all = {}
    for up in uploaded:
        # Handle zip
        if up.name.lower().endswith(".zip"):
            tmpdir = tempfile.mkdtemp()
            zpath = Path(tmpdir) / up.name
            with open(zpath, "wb") as f:
                f.write(up.getvalue())
            try:
                with zipfile.ZipFile(zpath, "r") as z:
                    z.extractall(tmpdir)
                # collect image files
                imgs = []
                for root, _, files in os.walk(tmpdir):
                    for f in files:
                        if is_image_filename(f):
                            imgs.append(Path(root) / f)
                if not imgs:
                    st.warning(f"No images found inside {up.name}")
                else:
                    st.info(f"Classifying {len(imgs)} images from {up.name} ...")
                    for imgp in imgs:
                        with open(imgp, "rb") as f:
                            b = f.read()
                        try:
                            res, pil = predict_from_bytes(model, b, class_map=class_map, top_k=top_k, use_vgg=use_vgg)
                            results_all[str(imgp)] = res
                            # display image at half size (improve look)
                            display_width = max(32, pil.width // 2)
                            st.image(pil, caption=f"{imgp.name} — Top: {res[0][1]} ({res[0][2]:.3f})", width=display_width)
                            for rank, (idx, name, p) in enumerate(res, start=1):
                                st.write(f"{rank}. {name} (idx {idx}) — {p:.4f}")
                            st.markdown("---")
                        except Exception as e:
                            st.error(f"Prediction failed for {imgp.name}: {e}")
            except Exception as e:
                st.error(f"Failed to extract or process zip {up.name}: {e}")

        # Single image file
        elif is_image_filename(up.name):
            b = up.getvalue()
            try:
                res, pil = predict_from_bytes(model, b, class_map=class_map, top_k=top_k, use_vgg=use_vgg)
                results_all[up.name] = res
                display_width = max(32, pil.width // 2)
                st.image(pil, caption=f"{up.name} — Top: {res[0][1]} ({res[0][2]:.3f})", width=display_width)
                for rank, (idx, name, p) in enumerate(res, start=1):
                    st.write(f"{rank}. {name} (idx {idx}) — {p:.4f}")
                st.markdown("---")
            except Exception as e:
                st.error(f"Prediction failed for {up.name}: {e}")
        else:
            st.warning(f"Skipping file {up.name} — unsupported type.")

    st.success("Done classifying uploaded files.")
    # Optionally show results as CSV download
    if st.button("Download results (CSV)"):
        import pandas as pd
        rows = []
        for fname, res in results_all.items():
            for rank, (idx, name, p) in enumerate(res, start=1):
                rows.append({"file": fname, "rank": rank, "idx": idx, "label": name, "prob": p})
        df = pd.DataFrame(rows)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, file_name="predictions.csv", mime="text/csv")

else:
    st.info("No files uploaded yet. Upload images or a zip to start predictions.")
