# streamlit_app.py
import os
import logging
import warnings
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageDraw, ImageFile
import numpy as np
import gc
import urllib.request
import streamlit as st
from scipy.special import softmax
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import Counter

# --- PIL safety for very large images ---
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- TensorFlow & logging setup ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

# ---- Streamlit page setup ----
st.set_page_config(
    page_title="Web Application for Sugarcane Age Detection using Drone Imagery",
    layout="wide"
)

# ---- Configuration ----
LOCAL_MODEL_FILENAME = "final_model_noopt.keras"
GITHUB_RAW_MODEL_URL = "https://raw.githubusercontent.com/pawarpm/sugar1/main/final_model_noopt.keras"
USE_VGG_PREPROCESS = False
TILE_SIZE = 224
DEFAULT_BATCH_SIZE = 64
MAX_TILE_THUMBNAILS = 40
OVERLAY_MAX_SIDE_DEFAULT = 3000

DEFAULT_CLASS_MAP = {
    0: "11_month",
    1: "2_month",
    2: "4_month",
    3: "6_month",
    4: "9_month"
}
LOGO_URL = "https://coe.sveri.ac.in/wp-content/themes/SVERICoE/images/sverilogo.png"

# ---- Utility Functions ----
def get_model_input_size(model):
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list):
        shape = shape[0]
    if not shape:
        return (240, 240, 3)
    if len(shape) == 4:
        _, h, w, c = shape
        return (int(h or 240), int(w or 240), int(c or 3))
    return (240, 240, 3)

def infer_num_classes_from_model(model):
    try:
        out_shape = model.output_shape
        if isinstance(out_shape, list): out_shape = out_shape[0]
        if isinstance(out_shape, tuple) and len(out_shape) >= 2:
            n = out_shape[-1]
            if isinstance(n, int): return n
    except Exception:
        pass
    try:
        for layer in reversed(model.layers):
            if hasattr(layer, "units"):
                n = getattr(layer, "units")
                if isinstance(n, int) and n > 1: return n
    except Exception:
        pass
    return None

def get_num_classes(model):
    n = infer_num_classes_from_model(model)
    return n if isinstance(n, int) and n > 1 else len(DEFAULT_CLASS_MAP)

def build_default_class_map(model):
    n = get_num_classes(model)
    return DEFAULT_CLASS_MAP.copy() if n == len(DEFAULT_CLASS_MAP) else {i: f"class_{i}" for i in range(n)}

def load_model_preferred(path, compile=False):
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(f"Model not found: {path}")
    return load_model(str(p), compile=compile)

@st.cache_resource(show_spinner=False)
def get_model_from_repo(local_name=LOCAL_MODEL_FILENAME, raw_url=GITHUB_RAW_MODEL_URL):
    local_path = Path(__file__).parent / local_name
    if local_path.exists():
        model = load_model_preferred(local_path, compile=False)
        return model, str(local_path)
    tmp_path = "/tmp/model.keras"
    urllib.request.urlretrieve(raw_url, tmp_path)
    model = load_model_preferred(tmp_path, compile=False)
    return model, tmp_path

def preprocess_tile_for_model(pil_img, target_hw, use_vgg=False):
    h, w = target_hw
    arr = np.array(pil_img.resize((w, h), Image.BILINEAR)).astype("float32")
    return arr / 255.0 if not use_vgg else tf.keras.applications.vgg16.preprocess_input(arr)

def predict_tiles_streaming(model, stitched_image, crop_size=224, batch_size=64, use_vgg=False):
    inp_h, inp_w, _ = get_model_input_size(model)
    width, height = stitched_image.size
    boxes = [(x, y, x + crop_size, y + crop_size)
             for y in range(0, height, crop_size)
             for x in range(0, width, crop_size)
             if x + crop_size <= width and y + crop_size <= height]
    n = len(boxes)
    num_classes = get_num_classes(model)
    probs_out = np.zeros((n, num_classes), dtype="float32")
    for i in range(0, n, batch_size):
        batch_boxes = boxes[i:i + batch_size]
        batch = np.stack([preprocess_tile_for_model(stitched_image.crop(box), (inp_h, inp_w), use_vgg) for box in batch_boxes])
        pred = model.predict(batch, verbose=0)
        if pred.sum(axis=1).max() > 1.0001 or pred.min() < 0:
            pred = softmax(pred, axis=1)
        probs_out[i:i + len(batch_boxes)] = pred
        del batch, pred
        gc.collect()
    return boxes, probs_out

def make_downscaled_overlay(base_img, boxes, labels, label_to_color, max_side=3000):
    W, H = base_img.size
    scale = min(1.0, max_side / max(W, H))
    base_small = base_img.resize((int(W * scale), int(H * scale)), Image.BILINEAR) if scale < 1.0 else base_img
    scaled_boxes = [(int(x0 * scale), int(y0 * scale), int(x1 * scale), int(y1 * scale)) for (x0, y0, x1, y1) in boxes]
    overlay = Image.new("RGBA", base_small.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for box, lbl in zip(scaled_boxes, labels):
        draw.rectangle(box, fill=label_to_color.get(lbl, (0, 0, 0, 120)))
    composited = Image.alpha_composite(base_small.convert("RGBA"), overlay)
    buf = BytesIO()
    composited.convert("RGB").save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()

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
st.markdown("""
**About the Model (brief):**
This application uses a **MobileNetV2 backbone** fine-tuned on drone imagery of sugarcane fields.
It classifies sugarcane crop age into stages such as *2, 4, 6, 9,* and *11 months*.
The model was trained using annotated drone datasets collected across multiple farms.
""")

# ---- Sidebar ----
with st.sidebar:
    st.header("Processing Settings")
    batch_size = st.slider("Batch size", 8, 128, DEFAULT_BATCH_SIZE, step=8)
    overlay_max_side = st.slider("Overlay max side (px)", 1000, 8000, OVERLAY_MAX_SIDE_DEFAULT, step=500)
    use_vgg = st.checkbox("Use VGG preprocessing (/255.0 off)", value=USE_VGG_PREPROCESS)

# ---- Load Model ----
with st.spinner("Loading model from repository..."):
    try:
        model, model_path = get_model_from_repo()
        st.success(f"✅ Model loaded successfully from: {model_path}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# ---- Class map ----
model_classes = infer_num_classes_from_model(model)
class_map = DEFAULT_CLASS_MAP if model_classes == len(DEFAULT_CLASS_MAP) else build_default_class_map(model)
st.info("Using default sugarcane age mapping." if class_map == DEFAULT_CLASS_MAP else "Default mapping size mismatch; using generic labels.")

# ---- Main App ----
st.header("Upload a single stitched farm image (JPEG/PNG)")
stitched_file = st.file_uploader("Upload stitched image (one file only)", type=["jpg", "jpeg", "png"])

if stitched_file:
    try:
        stitched_image = Image.open(stitched_file).convert("RGB")
        st.image(stitched_image, caption=f"Uploaded stitched image: {stitched_file.name}", width="stretch")
        st.write("---")
        st.write("### Analyzing image and classifying tiles...")
        boxes, probs = predict_tiles_streaming(model, stitched_image, TILE_SIZE, batch_size, use_vgg)

        if not boxes:
            st.warning("The image is smaller than 224×224 and could not be tiled.")
        else:
            predicted_indices = np.argmax(probs, axis=1)
            predicted_labels = [class_map.get(int(i), f"class_{i}") for i in predicted_indices]
            counts = Counter(predicted_labels)
            total_tiles = len(boxes)
            major_lbl, major_cnt = counts.most_common(1)[0]

            st.subheader("✅ Overall Prediction Summary")
            c1, c2 = st.columns(2)
            c1.metric("Final Predicted Age (Majority Vote)", major_lbl)
            c2.metric("Number of Tiles Analyzed", total_tiles)

            st.write("#### Field Composition by Predicted Age:")
            for lbl, cnt in counts.items():
                st.write(f"- **{lbl}:** {cnt} tiles — **{(cnt/total_tiles)*100:.2f}%**")

            st.write("---")

            # Overlay Map
            palette = [(31,119,180,140),(255,127,14,140),(44,160,44,140),
                       (214,39,40,140),(148,103,189,140),(140,86,75,140)]
            label_to_color = {lbl: palette[i % len(palette)] for i, lbl in enumerate(sorted(set(class_map.values())))}
            overlay_jpeg = make_downscaled_overlay(stitched_image, boxes, predicted_labels, label_to_color, overlay_max_side)

            st.subheader("Spatial Overlay Map (tiles colored by predicted class)")
            st.image(overlay_jpeg, caption="Overlay: semi-transparent tile predictions", width="stretch")

            st.write("#### Legend:")
            legend_cols = st.columns(len(label_to_color))
            for i, (lbl, color) in enumerate(label_to_color.items()):
                with legend_cols[i]:
                    sw = Image.new("RGBA", (50, 30), color)
                    buf = BytesIO(); sw.save(buf, format="PNG")
                    st.image(buf.getvalue(), width=60)
                    pct = (counts.get(lbl, 0)/total_tiles)*100
                    st.markdown(f"**{lbl}**  \n{pct:.2f}%")

            st.write("---")
            st.subheader("Sample Tile Analysis")
            cols = st.columns(4)
            for i in range(min(MAX_TILE_THUMBNAILS, total_tiles)):
                crop = stitched_image.crop(boxes[i])
                label = predicted_labels[i]
                conf = float(np.max(probs[i]))
                with cols[i % 4]:
                    st.image(crop, caption=f"Tile #{i+1}", width=min(200, crop.width // 2))
                    st.success(f"{label} ({conf:.3f})")

            del probs, overlay_jpeg
            gc.collect()

    except Exception as e:
        st.error("An error occurred during processing:")
        st.exception(e)

else:
    st.info("Please upload a stitched image (JPEG/PNG) to start classification.")

# ---- Footer ----
st.markdown("---")
st.markdown("""
**Project PI / Contact:**  
Dr. Prashant Maruti Pawar  
SVERI's College of Engineering, Pandharpur  
For collaboration or data access, please contact the institute.  
v6 — Optimized (no CSV storage, low-memory overlay)
""")
