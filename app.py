import streamlit as st
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

# ── Load TFLite Model ──
@st.cache_resource
def load_model():
    model_path = "drowsiness_model.tflite"
    if not os.path.exists(model_path):
        with st.spinner("Model download ho raha hai... pehli baar thoda waqt lagega ⏳"):
            gdown.download(
                id="1MgbhNZlVhwDfqtrU3zX8E9CVClLSjdB0",
                output=model_path,
                quiet=False
            )
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=model_path)
    except:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict(image, interpreter):
    img = image.convert("RGB").resize((128, 128))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()

    prob = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
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
    data = {
        "Model": ["MobileNetV2", "EfficientNetB0 ✅"],
        "Accuracy": ["85.56%", "88.44%"],
        "F1-Score": ["85.43%", "88.44%"]
    }
    st.table(data)

# ── Load Model ──
with st.spinner("🔄 AI Model load ho raha hai..."):
    interpreter = load_model()
st.success("✅ Model ready!")
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
                label, confidence = predict(image, interpreter)
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
                label, confidence = predict(image, interpreter)
            if "Alert" in label:
                st.success("✅ ALERT — Eyes Open!")
            else:
                st.error("😴 DROWSY — Wake Up!")
            st.metric("Prediction",  label)
            st.metric("Confidence", f"{confidence*100:.1f}%")
            st.progress(confidence)

st.divider()
st.markdown("""
<center>
🚗 Driver Drowsiness Detection | EfficientNetB0 | Accuracy: 88.44%
</center>
""", unsafe_allow_html=True)
