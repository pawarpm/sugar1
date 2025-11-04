# streamlit_app.py
import os
import logging
import warnings
from pathlib import Path
from io import BytesIO, StringIO
from PIL import Image, ImageDraw, ImageFile
import numpy as np

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
from collections import Counter
import urllib.request
import gc

# Snowflake (optional) – imported lazily only if user enables saving
try:
    from snowflake.snowpark.context import get_active_session  # noqa
    SNOWFLAKE_AVAILABLE = True
except Exception:
    SNOWFLAKE_AVAILABLE = False

# ---- Streamlit config ----
st.set_page_config(
    page_title="Web Application for Sugarcane Age Detection using Drone Imagery",
    layout="wide"
)

# ---- Configuration ----
LOCAL_MODEL_FILENAME = "final_model_noopt.keras"
GITHUB_RAW_MODEL_URL = "https://raw.githubusercontent.com/pawarpm/sugar1/main/final_model_noopt.keras"

USE_VGG_PREPROCESS = False
TOP_K_DEFAULT = 3
TILE_SIZE = 224                  # confirmed working crop size
DEFAULT_BATCH_SIZE = 64          # safe default; adjustable in sidebar
MAX_TILE_THUMBNAILS = 40         # cap thumbnails for UI performance
OVERLAY_MAX_SIDE_DEFAULT = 3000  # downscaled overlay longest side (pixels)

DEFAULT_CLASS_MAP = {0:"11_month",1:"2_month",2:"4_month",3:"6_month",4:"9_month"}
LOGO_URL = "https://coe.sveri.ac.in/wp-content/themes/SVERICoE/images/sverilogo.png"

# ---- Utilities ----
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

def get_num_classes(model):
    n = infer_num_classes_from_model(model)
    if isinstance(n, int) and n > 1:
        return n
    return len(DEFAULT_CLASS_MAP)

def build_default_class_map(model, prefix="class_"):
    n = get_num_classes(model)
    if n == len(DEFAULT_CLASS_MAP):
        return DEFAULT_CLASS_MAP.copy()
    return {i: f"{prefix}{i}" for i in range(n)}

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
def get_model_from_repo(local_name: str = LOCAL_MODEL_FILENAME, raw_url: str = GITHUB_RAW_MODEL_URL):
    """
    Prefer local repo file; if missing, fetch from GitHub raw to /tmp and load.
    """
    local_path = Path(__file__).parent / local_name
    if local_path.exists():
        model = load_model_preferred(str(local_path), compile=False)
        return model, str(local_path)
    tmp_path = "/tmp/model.keras"
    try:
        urllib.request.urlretrieve(raw_url, tmp_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download model from GitHub: {e}")
    model = load_model_preferred(tmp_path, compile=False)
    return model, tmp_path

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

def predict_tiles_streaming(model, stitched_image, crop_size=224, batch_size=64, use_vgg=False):
    """
    Predict tiles in batches to avoid large memory usage.
    Returns:
      boxes: list of (x0,y0,x1,y1)
      probs_out: np.ndarray of shape (N, num_classes)
    """
    inp_h, inp_w, _ = get_model_input_size(model)
    width, height = stitched_image.size
    boxes = []
    for y in range(0, height, crop_size):
        for x in range(0, width, crop_size):
            if x + crop_size <= width and y + crop_size <= height:
                boxes.append((x, y, x + crop_size, y + crop_size))
    n = len(boxes)
    num_classes = get_num_classes(model)
    probs_out = np.zeros((n, num_classes), dtype="float32")
    # Batch loop
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
        try:
            if pred.sum(axis=1).max() > 1.0001 or pred.min() < 0:
                pred = softmax(pred, axis=1)
        except Exception:
            pass
        probs_out[i:i+len(batch_boxes), :] = pred
        # free chunk asap
        del batch_arrs, batch, pred
        gc.collect()
    return boxes, probs_out

def make_downscaled_overlay(base_img, boxes, labels, label_to_color, max_side=3000):
    """
    Build semi-transparent overlay on a downscaled copy to reduce memory.
    Returns composited JPEG bytes.
    """
    W, H = base_img.size
    scale = min(1.0, max_side / max(W, H))
    if scale < 1.0:
        base_small = base_img.resize((int(W * scale), int(H * scale)), Image.BILINEAR)
        scaled_boxes = [(int(x0*scale), int(y0*scale), int(x1*scale), int(y1*scale)) for (x0,y0,x1,y1) in boxes]
    else:
        base_small = base_img
        scaled_boxes = boxes

    overlay = Image.new("RGBA", base_small.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for box, lbl in zip(scaled_boxes, labels):
        draw.rectangle(box, fill=label_to_color.get(lbl, (0, 0, 0, 120)), outline=None)

    composited = Image.alpha_composite(base_small.convert("RGBA"), overlay)
    # compress to JPEG to reduce display memory + speed up transfer
    buf = BytesIO()
    composited.convert("RGB").save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()

def try_save_to_snowflake_stage(bytes_data, stage_path):
    """
    Save bytes to a Snowflake stage using put_stream.
    Requires running inside Snowflake with snowpark available.
    """
    if not SNOWFLAKE_AVAILABLE:
        raise RuntimeError("Snowflake Snowpark not available in this environment.")
    session = get_active_session()
    from io import BytesIO
    # Ensures directory segment exists; Snowflake stages are object stores, path semantics are straightforward.
    session.file.put_stream(BytesIO(bytes_data), stage_path, overwrite=True)

# ---- Header ----
col1, col2 = st.columns([1, 5])
with col1:
    st.image(LOGO_URL, width=130)
with col2:
    st.markdown("## Web Application for Sugarcane Age Detection using Drone Imagery")
    st.markdown("**Developed by:** SVERI's College of Engineering, Pandharpur  ")
    st.markdown("**Research funding support from:** Rajiv Gandhi Science and Technology Commission, Government of Maharashtra")

st.markdown("---")

# ---- About ----
st.markdown(
    """
**About the Model (brief):**

This application uses a **MobileNetV2 backbone** fine-tuned on drone imagery of sugarcane fields.
It classifies sugarcane crop age into stages such as *2, 4, 6, 9,* and *11 months*.
The final layer is a dense classification head using Softmax activation.
The model was trained using annotated drone datasets collected across multiple farms.
"""
)

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Performance & Output")
    batch_size = st.slider("Batch size", min_value=8, max_value=128, value=DEFAULT_BATCH_SIZE, step=8)
    overlay_max_side = st.slider("Overlay max side (px)", min_value=1000, max_value=8000, value=OVERLAY_MAX_SIDE_DEFAULT, step=500)
    use_vgg = st.checkbox("Use VGG preprocessing (/255.0 off)", value=USE_VGG_PREPROCESS)

    st.divider()
    st.subheader("Save Outputs to Snowflake (optional)")
    save_to_stage = st.checkbox("Save overlay & CSV to Snowflake stage", value=False)
    stage_overlay_path = st.text_input("Stage path for overlay JPEG", "@APP_STAGE/exports/overlay.jpg")
    stage_csv_path = st.text_input("Stage path for CSV", "@APP_STAGE/exports/tiles.csv")
    if save_to_stage and not SNOWFLAKE_AVAILABLE:
        st.warning("Snowflake Snowpark not detected here. Enable this only inside Streamlit in Snowflake.")

# ---- Load model ----
with st.spinner("Loading model from repository..."):
    try:
        model, model_path = get_model_from_repo()
        st.success(f"✅ Model loaded from: {model_path}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# ---- Class map ----
model_classes = infer_num_classes_from_model(model)
if model_classes == len(DEFAULT_CLASS_MAP):
    class_map = DEFAULT_CLASS_MAP
    st.info("Using default sugarcane age mapping.")
else:
    class_map = build_default_class_map(model)
    st.warning("Default mapping size mismatch; using generic labels.")

# ---- Stitched image flow ----
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
        st.write("### Converting stitched image into 224×224 crops and classifying tiles...")

        try:
            boxes, probs = predict_tiles_streaming(
                model, stitched_image, crop_size=TILE_SIZE, batch_size=batch_size, use_vgg=use_vgg
            )

            if len(boxes) == 0:
                st.warning("The stitched image is smaller than 224×224 and could not be tiled.")
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

                # ---- Legend colors (deterministic) ----
                palette = [
                    (31,119,180,140),(255,127,14,140),(44,160,44,140),
                    (214,39,40,140),(148,103,189,140),(140,86,75,140),
                    (227,119,194,140),(127,127,127,140),
                ]
                label_to_color = {}
                for i, lbl in enumerate(sorted(list(set(class_map.values())))):
                    label_to_color[lbl] = palette[i % len(palette)]

                # ---- Downscaled overlay (memory-friendly) ----
                overlay_jpeg_bytes = make_downscaled_overlay(
                    stitched_image, boxes, predicted_labels, label_to_color, max_side=overlay_max_side
                )
                st.subheader("Spatial Overlay Map (tiles colored by predicted class)")
                st.image(overlay_jpeg_bytes, caption="Overlay: semi-transparent tile predictions", width="stretch")

                # ---- Legend & percentages ----
                st.write("#### Legend and Percentages")
                legend_cols = st.columns(len(label_to_color))
                for i, (lbl, color) in enumerate(label_to_color.items()):
                    with legend_cols[i]:
                        sw = Image.new("RGBA", (50, 30), color)
                        # small PNG here is fine
                        buf_sw = BytesIO()
                        sw.save(buf_sw, format="PNG")
                        st.image(buf_sw.getvalue(), width=60)
                        cnt = counts.get(lbl, 0)
                        pct = (cnt/total_tiles)*100 if total_tiles>0 else 0.0
                        st.markdown(f"**{lbl}**  \n{cnt} tiles  \n{pct:.2f}%")

                st.write("---")

                # ---- Sample thumbnails (kept small to save memory) ----
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

                # ---- CSV (pandas-free) ----
                rows = []
                for i, (box, pi) in enumerate(zip(boxes, probs), start=1):
                    x0, y0, x1, y1 = box
                    pred_idx = int(np.argmax(pi))
                    pred_label = class_map.get(pred_idx, f"class_{pred_idx}")
                    rows.append({
                        "tile_id": i,
                        "x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1,
                        "predicted_label": pred_label, "probability": float(np.max(pi))
                    })

                import io, csv
                csv_buffer = io.StringIO()
                fieldnames = ["tile_id", "x_min", "y_min", "x_max", "y_max", "predicted_label", "probability"]
                writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

                st.download_button(
                    label="Download Results (CSV)",
                    data=csv_buffer.getvalue().encode("utf-8"),
                    file_name="stitched_predictions.csv",
                    mime="text/csv",
                )

                # ---- Optional: Save overlay & CSV to Snowflake stage ----
                if save_to_stage:
                    try:
                        try_save_to_snowflake_stage(overlay_jpeg_bytes, stage_overlay_path)
                        try_save_to_snowflake_stage(csv_buffer.getvalue().encode("utf-8"), stage_csv_path)
                        st.success(f"Saved overlay to {stage_overlay_path} and CSV to {stage_csv_path}")
                    except Exception as e:
                        st.error(f"Failed to save to Snowflake stage: {e}")

                # Free big arrays sooner
                del probs, overlay_jpeg_bytes
                gc.collect()

        except Exception as e:
            st.error("An error occurred during tiling or prediction.")
            st.exception(e)

else:
    st.info("Please upload a single stitched image (JPEG/PNG) to begin classification.")

# ---- Footer ----
st.markdown("---")
st.markdown(
    """
**Project PI / Contact:**  
Dr. Prashant Maruti Pawar  
SVERI's College of Engineering, Pandharpur  
For collaboration or data access, please contact the institute.
v5 04nov2025 (Downscaled overlay + optional Snowflake save)
"""
)
