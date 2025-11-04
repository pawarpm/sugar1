# streamlit_app.py
import os
import logging
import warnings
from pathlib import Path
from io import BytesIO, StringIO
from PIL import Image, ImageDraw, ImageFile
import numpy as np
import tempfile

# --- PIL safety for very large images ---
Image.MAX_IMAGE_PIXELS = None                # allow very large stitched images
ImageFile.LOAD_TRUNCATED_IMAGES = True       # load even if slightly truncated

# Suppress noisy logs/warnings before importing TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # hide INFO/WARNING/ERROR from TF C++ logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    # force CPU; prevents CUDA init attempts
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import streamlit as st
from scipy.special import softmax
import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown
from collections import Counter

# Also ensure TF doesn't try GPU even if present
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

# ---- Streamlit config (unchanged) ----
st.set_page_config(
    page_title="Web Application for Sugarcane Age Detection using Drone Imagery",
    layout="wide"
)

# ---- Configuration (unchanged) ----
DRIVE_FILE_ID_DEFAULT = "10JYTIb9CWNhGbhnBNEA1Yj8SVVqx5BjE"
DEFAULT_MODEL_FILENAME = "/tmp/model.keras"
VALID_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
USE_VGG_PREPROCESS = False
TOP_K_DEFAULT = 3
TILE_SIZE = 160             # crop size
BATCH_SIZE = 64             # safe default; adjust if needed
MAX_TILE_THUMBNAILS = 40    # limit thumbnails in grid to keep UI snappy

DEFAULT_CLASS_MAP = {0:"11_month",1:"2_month",2:"4_month",3:"6_month",4:"9_month"}
LOGO_URL = "https://coe.sveri.ac.in/wp-content/themes/SVERICoE/images/sverilogo.png"

# ---- Utils (unchanged logic) ----
def get_model_input_size(model):
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list):
        shape = shape[0]
    if not shape:
        return (240, 240, 3)
    if len(shape) == 4:
        _, h, w, c = shape
        return (int(h) if h else 240, int(w) if w else 240, int(c) if c else 3)
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
    return {} if n is None else {i: f"{prefix}{i}" for i in range(n)}

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

def preprocess_tile_for_model(pil_img, target_hw, use_vgg=False):
    h, w = target_hw
    img = pil_img.resize((w, h), Image.BILINEAR)
    arr = np.array(img).astype("float32")
    if use_vgg:
        from tensorflow.keras.applications.vgg16 import preprocess_input
        arr = preprocess_input(arr)
    else:
        arr = arr / 255.0
    return arr

def predict_tiles_streaming(model, stitched_image, crop_size=160, batch_size=64, use_vgg=False):
    """Yield probs for tiles without storing all tiles in memory."""
    inp_h, inp_w, _ = get_model_input_size(model)
    width, height = stitched_image.size

    # Precompute all full-tile boxes
    boxes = []
    for y in range(0, height, crop_size):
        for x in range(0, width, crop_size):
            if x + crop_size <= width and y + crop_size <= height:
                boxes.append((x, y, x + crop_size, y + crop_size))

    n = len(boxes)
    probs_out = np.zeros((n, infer_num_classes_from_model(model)), dtype="float32")
    # Process in batches
    for i in range(0, n, batch_size):
        batch_boxes = boxes[i:i+batch_size]
        batch_arrs = []
        for box in batch_boxes:
            crop = stitched_image.crop(box)
            arr = preprocess_tile_for_model(crop, (inp_h, inp_w), use_vgg=use_vgg)
            batch_arrs.append(arr)
        batch = np.stack(batch_arrs, axis=0)
        pred = model.predict(batch, verbose=0)
        pred = pred[0] if (hasattr(pred, "ndim") and pred.ndim == 3 and pred.shape[0] == 1) else pred
        # force probs if needed
        try:
            if pred.sum(axis=1).max() > 1.0001 or pred.min() < 0:
                pred = softmax(pred, axis=1)
        except Exception:
            pass
        probs_out[i:i+len(batch_boxes), :] = pred

    return boxes, probs_out

# ---- Header (unchanged) ----
col1, col2 = st.columns([1, 5])
with col1:
    st.image(LOGO_URL, width=130)
with col2:
    st.markdown("## Web Application for Sugarcane Age Detection using Drone Imagery")
    st.markdown("**Developed by:** SVERI's College of Engineering, Pandharpur  ")
    st.markdown("**Research funding support from:** Rajiv Gandhi Science and Technology Commission, Government of Maharashtra")

st.markdown("---")

# ---- About (unchanged) ----
st.markdown(
    """
**About the Model (brief):**

This application uses a **MobileNetV2 backbone** fine-tuned on drone imagery of sugarcane fields.
It classifies sugarcane crop age into stages such as *2, 4, 6, 9,* and *11 months*.
The final layer is a dense classification head using Softmax activation.
The model was trained using annotated drone datasets collected across multiple farms.
"""
)

# ---- Sidebar (unchanged) ----
with st.sidebar:
    st.header("Model / Prediction Settings")
    drive_id = st.text_input("Google Drive File ID", value=DRIVE_FILE_ID_DEFAULT)
    model_dest = st.text_input("Local Model Path", value=DEFAULT_MODEL_FILENAME)
    force_dl = st.checkbox("Force re-download model", value=False)
    top_k = st.number_input("Top K predictions", min_value=1, max_value=10, value=TOP_K_DEFAULT)
    use_vgg = st.checkbox("Use VGG preprocessing (/255.0 off)", value=USE_VGG_PREPROCESS)

# ---- Load model (unchanged) ----
with st.spinner("Downloading and loading model..."):
    try:
        model, model_path = get_model_from_drive(drive_file_id=drive_id, local_path=model_dest, force=force_dl)
        st.success(f"✅ Model loaded successfully from: {model_path}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# ---- Class map (unchanged) ----
model_classes = infer_num_classes_from_model(model)
class_map = DEFAULT_CLASS_MAP if (model_classes == len(DEFAULT_CLASS_MAP)) else build_default_class_map(model)
st.info("Using default sugarcane age mapping." if class_map == DEFAULT_CLASS_MAP else "Default mapping size mismatch; using generic labels.")

# ---- Stitched image flow (streaming, memory-safe) ----
st.header("Upload a single stitched farm image (JPEG/PNG)")
stitched_file = st.file_uploader("Upload stitched image (one file only)", accept_multiple_files=False, type=["jpg", "jpeg", "png"])

if stitched_file is not None:
    try:
        stitched_image = Image.open(stitched_file).convert("RGB")
    except Exception as e:
        st.error(f"Failed to open uploaded image: {e}")
        stitched_image = None

    if stitched_image is not None:
        st.image(stitched_image, caption=f"Uploaded stitched image: {stitched_file.name}", width="stretch")
        st.write("---")
        st.write("### Tiling stitched image into 160x160 crops and classifying tiles...")

        try:
            # Streamed tile prediction (no giant arrays)
            crop_size = TILE_SIZE
            boxes, probs = predict_tiles_streaming(
                model, stitched_image, crop_size=crop_size, batch_size=BATCH_SIZE, use_vgg=use_vgg
            )

            if len(boxes) == 0:
                st.warning("The stitched image is smaller than 160x160 and could not be tiled.")
            else:
                predicted_indices = np.argmax(probs, axis=1)
                predicted_labels = [class_map.get(int(idx), f"class_{idx}") for idx in predicted_indices]

                # Count and percentage
                counts = Counter(predicted_labels)
                total_tiles = len(boxes)

                st.subheader("✅ Overall Prediction Summary")
                c1, c2 = st.columns(2)
                (major_lbl, major_cnt) = counts.most_common(1)[0]
                with c1: st.metric("Final Predicted Age (Majority Vote)", major_lbl)
                with c2: st.metric("Number of Tiles Analyzed", total_tiles)

                st.write("#### Prediction Breakdown (tile counts and percentage of field):")
                for lbl, cnt in counts.items():
                    st.write(f"- **{lbl}:** {cnt} tiles — **{(cnt/total_tiles)*100:.2f}%** of field")

                st.write("---")

                # ---- Overlay map ----
                overlay = Image.new("RGBA", stitched_image.size, (0,0,0,0))
                draw = ImageDraw.Draw(overlay)
                palette = [
                    (31,119,180,140),(255,127,14,140),(44,160,44,140),
                    (214,39,40,140),(148,103,189,140),(140,86,75,140),
                    (227,119,194,140),(127,127,127,140),
                ]
                label_to_color = {}
                for i, lbl in enumerate(sorted(list(set(class_map.values())))):
                    label_to_color[lbl] = palette[i % len(palette)]
                for box, lbl in zip(boxes, predicted_labels):
                    draw.rectangle(box, fill=label_to_color.get(lbl, (0,0,0,120)), outline=None)

                composited = Image.alpha_composite(stitched_image.convert("RGBA"), overlay)
                st.subheader("Spatial Overlay Map (tiles colored by predicted class)")
                st.image(composited, caption="Overlay: semi-transparent tile predictions", width="stretch")

                # Legend
                st.write("#### Legend and Percentages")
                legend_cols = st.columns(len(label_to_color))
                for i, (lbl, color) in enumerate(label_to_color.items()):
                    with legend_cols[i]:
                        sw = Image.new("RGBA", (50, 30), color)
                        st.image(sw, width=60)
                        cnt = counts.get(lbl, 0)
                        pct = (cnt/total_tiles)*100 if total_tiles>0 else 0.0
                        st.markdown(f"**{lbl}**  \n{cnt} tiles  \n{pct:.2f}%")

                st.write("---")

                # Individual tiles – show a capped sample to keep UI light
                st.subheader("Individual Tile Analysis (sample)")
                cols = st.columns(4)
                sample_n = min(MAX_TILE_THUMBNAILS, total_tiles)
                for i in range(sample_n):
                    box = boxes[i]
                    crop = stitched_image.crop(box)
                    pred_idx = predicted_indices[i]
                    pred_label = class_map.get(int(pred_idx), f"class_{int(pred_idx)}")
                    confidence = float(np.max(probs[i]))
                    col = cols[i % 4]
                    with col:
                        st.image(crop, caption=f"Tile #{i+1}", width=min(200, max(64, crop.width // 2)))
                        st.success(f"Prediction: {pred_label} ({confidence:.3f})")

                # CSV
                rows = []
                for i, (box, pi) in enumerate(zip(boxes, probs), start=1):
                    x0,y0,x1,y1 = box
                    pred_idx = int(np.argmax(pi))
                    pred_label = class_map.get(pred_idx, f"class_{pred_idx}")
                    rows.append({
                        "tile_id": i, "x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1,
                        "predicted_label": pred_label, "probability": float(np.max(pi))
                    })
                if st.button("Download Results (CSV)"):
                    import pandas as pd
                    df = pd.DataFrame(rows)
                    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                                       "stitched_predictions.csv", "text/csv")
        except Exception as e:
            # Show any runtime error cleanly in UI
            st.error("An error occurred during tiling or prediction.")
            st.exception(e)

else:
    st.info("Please upload a single stitched image (JPEG/PNG) to begin classification.")

# ---- Footer (unchanged) ----
st.markdown("---")
st.markdown(
    """
**Project PI / Contact:**  
Dr. Prashant Maruti Pawar  
SVERI's College of Engineering, Pandharpur  
For collaboration or data access, please contact the institute.
v4 04nov2025
"""
)
