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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide INFO/WARNING from TF
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import streamlit as st
from scipy.special import softmax
import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown  # download shared file from Google Drive

# Streamlit config
st.set_page_config(
    page_title="Web Application for Sugarcane Age Detection using Drone Imagery (V2)",
    layout="wide"
)

# -------- Configuration --------
DRIVE_FILE_ID_DEFAULT = "10JYTIb9CWNhGbhnBNEA1Yj8SVVqx5BjE"
DEFAULT_MODEL_FILENAME = "/tmp/model.keras"
VALID_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
USE_VGG_PREPROCESS = False
TOP_K_DEFAULT = 3

# Default mapping
DEFAULT_CLASS_MAP = {
    0: "11_month",
    1: "2_month",
    2: "4_month",
    3: "6_month",
    4: "9_month"
}

# SVERI Logo URL
LOGO_URL = "https://coe.sveri.ac.in/wp-content/themes/SVERICoE/images/sverilogo.png"

# -------- Utility Functions --------
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

def sanitize_mapping(raw_map):
    if not isinstance(raw_map, dict):
        return None
    out = {}
    for k, v in raw_map.items():
        try:
            ik = int(k)
        except Exception:
            return None
        out[int(ik)] = str(v)
    return out

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
    probs = softmax(preds) if preds.sum() > 1.0001 or preds.min() < 0 else preds
    top_idx = probs.argsort()[-top_k:][::-1]
    results = [(int(idx), class_map.get(int(idx), f"class_{idx}"), float(probs[idx])) for idx in top_idx]
    return results, pil_img

def download_from_gdrive(file_id: str, dest_path: str, force=False):
    dest = Path(dest_path)
    if dest.exists() and not force:
        return str(dest)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(dest), quiet=False)
    return str(dest)

def load_model_preferred(path, convert_h5_to_keras=True, compile=False):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    if p.suffix.lower() == ".keras":
        return load_model(str(p), compile=compile)
    elif p.suffix.lower() == ".h5" and convert_h5_to_keras:
        model = load_model(str(p), compile=compile)
        keras_path = str(p.with_suffix(".keras"))
        model.save(keras_path)
        return load_model(keras_path, compile=compile)
    else:
        return load_model(str(p), compile=compile)

@st.cache_resource(show_spinner=False)
def get_model_from_drive(drive_file_id=DRIVE_FILE_ID_DEFAULT, local_path=DEFAULT_MODEL_FILENAME, force=False):
    dest = download_from_gdrive(drive_file_id, local_path, force=force)
    model = load_model_preferred(dest, compile=False)
    return model, dest

# -------- Streamlit UI --------
# Header section with logo and title
col1, col2 = st.columns([1, 5])
with col1:
    st.image(LOGO_URL, width=130)

with col2:
    st.markdown("## Web Application for Sugarcane Age Detection using Drone Imagery(V2)")
    st.markdown("**Developed by:** SVERI's College of Engineering, Pandharpur  ")
    st.markdown("**Research funding support from:** Rajiv Gandhi Science and Technology Commission, Government of Maharashtra")

st.markdown("---")

# About model
st.markdown(
    """
**About the Model (brief):**

This application uses a **MobileNetV2 backbone** fine-tuned on drone imagery of sugarcane fields.
It classifies sugarcane crop age into stages such as *2, 4, 6, 9,* and *11 months*.
The final layer is a dense classification head using Softmax activation.
The model was trained using annotated drone datasets collected across multiple farms.
"""
)

# Sidebar: model and settings
with st.sidebar:
    st.header("Model / Prediction Settings")
    drive_id = st.text_input("Google Drive File ID", value=DRIVE_FILE_ID_DEFAULT)
    model_dest = st.text_input("Local Model Path", value=DEFAULT_MODEL_FILENAME)
    force_dl = st.checkbox("Force re-download model", value=False)
    top_k = st.number_input("Top K predictions", min_value=1, max_value=10, value=TOP_K_DEFAULT)
    use_vgg = st.checkbox("Use VGG preprocessing (/255.0 off)", value=USE_VGG_PREPROCESS)

# Load model
with st.spinner("Downloading and loading model..."):
    try:
        model, model_path = get_model_from_drive(drive_file_id=drive_id, local_path=model_dest, force=force_dl)
        st.success(f"✅ Model loaded successfully from: {model_path}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# Apply class map automatically
model_classes = infer_num_classes_from_model(model)
if model_classes == len(DEFAULT_CLASS_MAP):
    class_map = DEFAULT_CLASS_MAP
    st.info("Using default sugarcane age mapping.")
else:
    class_map = build_default_class_map(model)
    st.warning("Default mapping size mismatch; using generic labels.")

# Upload section
st.header("Upload Images or ZIP Folder")
uploaded = st.file_uploader("Select one or more images, or a ZIP folder", accept_multiple_files=True, type=["jpg","jpeg","png","bmp","tiff","zip"])

if uploaded:
    # If user uploaded exactly one file and it is an image (not zip), treat it as stitched farm image and tile it
    if len(uploaded) == 1 and not uploaded[0].name.lower().endswith(".zip"):
        up = uploaded[0]
        try:
            stitched_image = Image.open(up).convert("RGB")
        except Exception as e:
            st.error(f"Failed to open uploaded image: {e}")
            stitched_image = None

        if stitched_image is not None:
            st.image(stitched_image, caption=f"Uploaded stitched image: {up.name}", width=700)
            st.write("---")
            st.write("### Tiling stitched image into 160x160 crops and classifying tiles...")

            crop_size = 160
            width, height = stitched_image.size
            cropped_images = []
            crop_boxes = []
            for y in range(0, height, crop_size):
                for x in range(0, width, crop_size):
                    if x + crop_size <= width and y + crop_size <= height:
                        box = (x, y, x + crop_size, y + crop_size)
                        crop = stitched_image.crop(box)
                        cropped_images.append(crop)
                        crop_boxes.append(box)

            if not cropped_images:
                st.warning("The stitched image is smaller than 160x160 and could not be tiled.")
            else:
                # Build batch array for prediction using same preprocess as single-image flow
                batch_list = []
                for crop in cropped_images:
                    arr = np.array(crop).astype("float32")
                    # normalize similarly to preprocess_image_for_model_bytes (resizing not needed; crops assumed same scale)
                    # But to be safe, resize crop to model input size:
                    inp_h, inp_w, inp_c = get_model_input_size(model)
                    crop_resized = crop.resize((inp_w, inp_h), Image.BILINEAR)
                    arr2 = np.array(crop_resized).astype("float32")
                    if use_vgg:
                        from tensorflow.keras.applications.vgg16 import preprocess_input
                        arr2 = preprocess_input(arr2)
                    else:
                        arr2 = arr2 / 255.0
                    batch_list.append(arr2)

                batch_array = np.stack(batch_list, axis=0)
                preds = model.predict(batch_array, verbose=0)
                # ensure preds is 2D (n_tiles, n_classes)
                preds = preds[0] if (hasattr(preds, "ndim") and preds.ndim == 3 and preds.shape[0] == 1) else preds

                # Convert to probabilities if needed
                try:
                    if preds.sum(axis=1).max() > 1.0001 or preds.min() < 0:
                        probs = softmax(preds, axis=1)
                    else:
                        probs = preds
                except Exception:
                    probs = preds

                predicted_indices = np.argmax(probs, axis=1)
                predicted_labels = [class_map.get(int(idx), f"class_{idx}") for idx in predicted_indices]

                # Count and percentage
                from collections import Counter
                counts = Counter(predicted_labels)
                total_tiles = len(cropped_images)

                st.subheader("✅ Overall Prediction Summary")
                col1, col2 = st.columns(2)
                most_common_label, most_common_count = counts.most_common(1)[0]
                with col1:
                    st.metric("Final Predicted Age (Majority Vote)", most_common_label)
                with col2:
                    st.metric("Number of Tiles Analyzed", total_tiles)

                st.write("#### Prediction Breakdown (tile counts and percentage of field):")
                for lbl, cnt in counts.items():
                    pct = (cnt / total_tiles) * 100
                    st.write(f"- **{lbl}:** {cnt} tiles — **{pct:.2f}%** of field")

                st.write("---")
                st.subheader("Individual Tile Analysis")
                num_columns = 4
                cols = st.columns(num_columns)
                for i, (crop, prob_row) in enumerate(zip(cropped_images, probs)):
                    col = cols[i % num_columns]
                    pred_idx = int(np.argmax(prob_row))
                    pred_label = class_map.get(pred_idx, f"class_{pred_idx}")
                    confidence = float(np.max(prob_row))
                    with col:
                        # Display smaller thumbnail for nicer layout
                        display_width = min(200, max(64, crop.width // 2))
                        st.image(crop, caption=f"Tile #{i+1}", width=display_width)
                        st.success(f"Prediction: {pred_label} ({confidence:.3f})")

                # Prepare results_all similar to previous behavior (for CSV)
                results_all = {}
                for i, prob_row in enumerate(probs):
                    pred_idx = int(np.argmax(prob_row))
                    pred_label = class_map.get(pred_idx, f"class_{pred_idx}")
                    results_all[f"{up.name}_tile_{i+1}"] = [(pred_idx, pred_label, float(prob_row[pred_idx]))]

                # Download CSV option (same format)
                if st.button("Download Results (CSV)"):
                    import pandas as pd
                    rows = []
                    for fname, res in results_all.items():
                        for rank, (idx, name, p) in enumerate(res, start=1):
                            rows.append({"file": fname, "rank": rank, "label": name, "probability": p})
                    df = pd.DataFrame(rows)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download CSV", csv, "predictions.csv", "text/csv")

    else:
        # Existing behavior for zip or multiple uploaded images (unchanged)
        results_all = {}
        for up in uploaded:
            if up.name.lower().endswith(".zip"):
                tmpdir = tempfile.mkdtemp()
                zpath = Path(tmpdir) / up.name
                with open(zpath, "wb") as f:
                    f.write(up.getvalue())
                with zipfile.ZipFile(zpath, "r") as z:
                    z.extractall(tmpdir)
                imgs = [str(x) for x in Path(tmpdir).rglob("*") if x.suffix.lower() in VALID_IMG_EXTS]
            else:
                imgs = [up]

            for img in imgs:
                try:
                    b = img.read() if hasattr(img, "read") else open(img, "rb").read()
                    res, pil = predict_from_bytes(model, b, class_map=class_map, top_k=top_k, use_vgg=use_vgg)
                    results_all[img] = res
                    display_width = min(400, max(64, pil.width // 2))
                    st.image(pil, caption=f"{Path(img).name} — Top: {res[0][1]} ({res[0][2]:.3f})", width=display_width)
                    for rank, (idx, name, p) in enumerate(res, start=1):
                        st.write(f"{rank}. {name} — {p:.4f}")
                    st.markdown("---")
                except Exception as e:
                    st.error(f"Prediction failed for {Path(img).name}: {e}")
        st.success("✅ Classification complete.")

        # Download results CSV
        if st.button("Download Results (CSV)"):
            import pandas as pd
            rows = []
            for fname, res in results_all.items():
                for rank, (idx, name, p) in enumerate(res, start=1):
                    rows.append({"file": Path(fname).name, "rank": rank, "label": name, "probability": p})
            df = pd.DataFrame(rows)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "predictions.csv", "text/csv")

else:
    st.info("Please upload images or a ZIP file to begin classification.")

# Footer
st.markdown("---")
st.markdown(
    """
**Project PI / Contact:**  
Dr. Prashant Maruti Pawar  
SVERI's College of Engineering, Pandharpur  
For collaboration or data access, please contact the institute.
"""
)
