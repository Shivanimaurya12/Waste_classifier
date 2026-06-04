import streamlit as st
import numpy as np
import cv2
import os
import pandas as pd
from PIL import Image
import tensorflow as tf

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Waste Classification System",
    page_icon="♻️",
    layout="centered"
)

# =========================
# SESSION STATE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

if "organic" not in st.session_state:
    st.session_state.organic = 0

if "recyclable" not in st.session_state:
    st.session_state.recyclable = 0

# =========================
# TITLE
# =========================
st.title("♻️ Waste Classification System")
st.write("AI-powered waste classification using TensorFlow Lite")

# =========================
# MODEL CONFIG
# =========================
MODEL_PATH = "model_int8.tflite"

# =========================
# LOAD TFLITE MODEL
# =========================
@st.cache_resource
def load_tflite_model():

    if not os.path.exists(MODEL_PATH):
        return None

    interpreter = tf.lite.Interpreter(
        model_path=MODEL_PATH
    )

    interpreter.allocate_tensors()

    return {
        "interpreter": interpreter,
        "input_details": interpreter.get_input_details(),
        "output_details": interpreter.get_output_details()
    }

model_data = load_tflite_model()

# =========================
# PREPROCESS
# =========================
def preprocess(img):

    img = cv2.resize(img, (224, 224))

    # Model expects uint8 values (0-255)
    img = img.astype(np.uint8)

    img = np.expand_dims(img, axis=0)

    return img

# =========================
# PREDICT
# =========================
def predict(img):

    if model_data is None:
        return "Model not loaded", 0.0

    interpreter = model_data["interpreter"]
    input_details = model_data["input_details"]
    output_details = model_data["output_details"]

    img = preprocess(img)

    interpreter.set_tensor(
        input_details[0]["index"],
        img
    )

    interpreter.invoke()

    pred = interpreter.get_tensor(
        output_details[0]["index"]
    )

    # Dequantize output
    scale, zero_point = output_details[0]["quantization"]

    score = (
        pred.astype(np.float32)[0][0]
        - zero_point
    ) * scale

    score = float(score)

    score = max(0.0, min(1.0, score))

    # Binary classifier
    if score >= 0.5:
        return "♻️ Recyclable Waste", score * 100
    else:
        return "🌿 Organic Waste", (1.0 - score) * 100

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("## ♻️ Waste Control")

page = st.sidebar.radio(
    "Navigation",
    ["🔍 Predict", "📊 Dashboard"]
)

# =========================
# PREDICTION PAGE
# =========================
if page == "🔍 Predict":

    st.subheader("Upload Waste Images")

    uploaded_files = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} image(s) uploaded"
        )

        cols = st.columns(3)

        for i, file in enumerate(uploaded_files):

            image = Image.open(file).convert("RGB")
            img_array = np.array(image)

            label, confidence = predict(img_array)

            # Update counters
            if "Organic" in label:
                st.session_state.organic += 1
            else:
                st.session_state.recyclable += 1

            # Save history
            st.session_state.history.append({
                "label": label,
                "confidence": confidence
            })

            with cols[i % 3]:

                st.image(
                    image,
                    use_container_width=True
                )

                st.success(label)

                st.progress(
                    float(confidence) / 100
                )

                st.write(
                    f"Confidence: {confidence:.2f}%"
                )

# =========================
# DASHBOARD PAGE
# =========================
elif page == "📊 Dashboard":

    st.subheader("Analytics Dashboard")

    total = (
        st.session_state.organic +
        st.session_state.recyclable
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Predictions",
        total
    )

    col2.metric(
        "Organic Waste",
        st.session_state.organic
    )

    col3.metric(
        "Recyclable Waste",
        st.session_state.recyclable
    )

    st.markdown("---")

    chart_data = pd.DataFrame({
        "Category": [
            "Organic",
            "Recyclable"
        ],
        "Count": [
            st.session_state.organic,
            st.session_state.recyclable
        ]
    })

    st.bar_chart(
        chart_data.set_index("Category")
    )

    st.markdown("---")

    st.subheader("Recent Predictions")

    if st.session_state.history:

        for item in st.session_state.history[-10:]:

            st.write(
                f"• {item['label']} "
                f"({item['confidence']:.2f}%)"
            )

    else:
        st.info("No predictions yet")

# =========================
# MODEL STATUS
# =========================
st.markdown("---")

if model_data is None:
    st.error(
        f"❌ {MODEL_PATH} not found."
    )
else:
    st.success(
        f"✅ {MODEL_PATH} loaded successfully."
    )

# =========================
# FOOTER
# =========================
st.markdown("""
---
🌍 Environmental Intelligence | 🤖 TensorFlow Lite Powered | 📈 Real-Time Analytics
""")