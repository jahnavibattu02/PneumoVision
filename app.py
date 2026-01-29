import json
import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image

from src.quality import quality_checks
from src.explain import find_last_conv_layer, make_gradcam_heatmap, overlay_heatmap_on_image, threshold_mask

st.set_page_config(page_title="PneumoVision", layout="centered")

st.title("PneumoVision 🩻")
st.caption("Educational demo only — not for clinical use.")

MODEL_PATH = "artifacts/model.keras"
LABELS_PATH = "artifacts/labels.json"

@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels_info = json.load(f)
    # labels_info["labels"] maps {index: class_name}
    labels = {int(k): v for k, v in labels_info["labels"].items()}
    return model, labels

def preprocess_for_model(img_bgr, img_size=224):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    x = img_resized.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    return x

model, labels = load_model_and_labels()
last_conv = find_last_conv_layer(model)

uploaded = st.file_uploader("Upload a chest X-ray image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    img_rgb = np.array(pil)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    st.subheader("Input")
    st.image(pil, use_container_width=True)

    st.subheader("Quality Check")
    qc = quality_checks(img_bgr)

    if qc["warnings"]:
        for w in qc["warnings"]:
            st.warning(w)
    else:
        st.success("Quality check passed ✅")

    st.caption(f"Stats: brightness={qc['stats'].get('mean_brightness', 0):.1f} | blur={qc['stats'].get('blur_score', 0):.1f} | contrast={qc['stats'].get('contrast', 0):.1f}")

    if not qc["passed"]:
        st.error("Prediction blocked due to failed quality gate.")
        st.stop()

    # Prediction
    x = preprocess_for_model(img_bgr, img_size=224)
    prob = float(model.predict(x, verbose=0).ravel()[0])
    pred = 1 if prob >= 0.5 else 0

    pred_label = labels.get(pred, str(pred))
    conf = prob if pred == 1 else (1 - prob)

    st.subheader("Prediction")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Label", pred_label)
    with col2:
        st.metric("Confidence", f"{conf*100:.1f}%")

    # Grad-CAM
    st.subheader("Explainability (Grad-CAM)")
    show_cam = st.toggle("Show Grad-CAM heatmap", value=True)
    alpha = st.slider("Overlay intensity", min_value=0.1, max_value=0.8, value=0.45, step=0.05)

    if show_cam:
        heatmap = make_gradcam_heatmap(model, x, last_conv)
        overlay, heatmap_resized = overlay_heatmap_on_image(img_bgr, heatmap, alpha=alpha)

        mask = threshold_mask(heatmap_resized, thresh=0.55)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        st.caption("Heatmap overlay (where the model focused).")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

        with st.expander("Show thresholded focus mask"):
            st.image(mask_rgb, use_container_width=True)

    st.info("Note: Grad-CAM is an interpretation aid, not a clinical explanation.")
else:
    st.caption("Tip: Use a clear frontal CXR image for best results.")
