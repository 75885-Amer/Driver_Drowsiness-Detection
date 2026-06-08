
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import gdown
import os

st.set_page_config(
    page_title="Driver Drowsiness Detector",
    page_icon="🚗",
    layout="centered"
)

st.markdown("""
<style>
body { background-color: #0f0f0f; }
.alert-box {
    background-color: #0a2e0a;
    border: 3px solid #2ecc71;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    font-size: 1.5rem;
    color: #2ecc71;
    font-weight: bold;
}
.drowsy-box {
    background-color: #2e0a0a;
    border: 3px solid #e74c3c;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    font-size: 1.5rem;
    color: #e74c3c;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = "efficientnetb0_best.h5"
    if not os.path.exists(model_path):
        with st.spinner("Model download ho raha hai... pehli baar thoda waqt lagega"):
            gdown.download(
                id="1MgbhNZlVhwDfqtrU3zX8E9CVClLSjdB0",
                output=model_path,
                quiet=False
            )
    return keras.models.load_model(model_path)

def predict(image, model):
    img = image.convert("RGB").resize((128, 128))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    prob = float(model.predict(arr, verbose=0)[0][0])
    label = "Open (Alert)" if prob > 0.5 else "Closed (Drowsy)"
    confidence = prob if prob > 0.5 else 1 - prob
    return label, confidence

# ── Header ──
st.title("🚗 Driver Drowsiness Detection")
st.markdown("**AI-Powered Real-Time Driver Safety System**")
st.divider()

# ── Sidebar ──
with st.sidebar:
    st.header("ℹ️ About Project")
    st.markdown("""
    **Best Model:** EfficientNetB0
    
    **Dataset:** MRL Eye Dataset
    
    **Total Images:** 6,000
    
    **Split:** 70% Train | 15% Val | 15% Test
    """)
    st.divider()
    st.header("📊 Model Results")
    st.metric("Test Accuracy",  "88.44%")
    st.metric("F1-Score",       "88.44%")
    st.metric("Precision",      "88.46%")
    st.metric("Recall",         "88.44%")
    st.metric("AUC-ROC",        "94.74%")
    st.divider()
    st.header("🏆 Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**MobileNetV2**")
        st.markdown("85.56%")
    with col2:
        st.markdown("**EfficientNetB0**")
        st.markdown("88.44% ✅")

# ── Load Model ──
with st.spinner("🔄 AI Model load ho raha hai..."):
    model = load_model()
st.success("✅ Model ready — detecting!")
st.divider()

# ── Tabs ──
tab1, tab2 = st.tabs(["📷 Camera", "📁 Upload Image"])

with tab1:
    st.markdown("### 📷 Camera Se Live Detection")
    camera_img = st.camera_input("Photo lo — model detect karega")
    if camera_img:
        image = Image.open(camera_img)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Captured", use_column_width=True)
        with col2:
            with st.spinner("Analyzing..."):
                label, confidence = predict(image, model)
            if "Alert" in label:
                st.markdown(f"""
                <div class="alert-box">
                    ✅ ALERT<br>Eyes Open<br>
                    <small>Confidence: {confidence*100:.1f}%</small>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="drowsy-box">
                    😴 DROWSY!<br>Wake Up!<br>
                    <small>Confidence: {confidence*100:.1f}%</small>
                </div>
                """, unsafe_allow_html=True)
                st.error("⚠️ WARNING: Driver is Drowsy!")

with tab2:
    st.markdown("### 📁 Image Upload Kar Ke Detection")
    uploaded = st.file_uploader(
        "Eye image upload karo (JPG/PNG)",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded:
        image = Image.open(uploaded)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        with col2:
            with st.spinner("Analyzing..."):
                label, confidence = predict(image, model)
            if "Alert" in label:
                st.success(f"✅ ALERT — Eyes Open")
            else:
                st.error(f"😴 DROWSY — Wake Up!")
            st.metric("Prediction",  label)
            st.metric("Confidence", f"{confidence*100:.1f}%")
            st.progress(confidence)

st.divider()
st.markdown("""
<center>
🚗 Driver Drowsiness Detection System<br>
EfficientNetB0 | MRL Eye Dataset | Test Accuracy: 88.44%
</center>
""", unsafe_allow_html=True)
