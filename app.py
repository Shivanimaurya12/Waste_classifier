import streamlit as st
import numpy as np
import cv2
import os
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model

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
st.write("AI-powered waste classification using Deep Learning")

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "model.h5"

@st.cache_resource
def load_model_safe():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    return None

model = load_model_safe()

# =========================
# PREPROCESS
# =========================
def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# PREDICTION (SAFE + CLEAN)
# =========================
def predict(img):
    if model is None:
        return "Model not loaded", 0

    img = preprocess(img)
    pred = model.predict(img, verbose=0)

    # Binary model (sigmoid)
    if pred.shape[-1] == 1:
        score = float(pred[0][0])

        if score >= 0.5:
            st.session_state.recyclable += 1
            return "♻️ Recyclable Waste", score * 100
        else:
            st.session_state.organic += 1
            return "🌿 Organic Waste", (1 - score) * 100

    # Multi-class model (softmax)
    else:
        class_id = int(np.argmax(pred))
        confidence = float(np.max(pred)) * 100

        if class_id == 0:
            st.session_state.organic += 1
            return "🌿 Organic Waste", confidence
        else:
            st.session_state.recyclable += 1
            return "♻️ Recyclable Waste", confidence

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("## ♻️ Waste Control")

page = st.sidebar.radio(
    "Navigation",
    ["🔍 Predict", "📊 Dashboard"]
)

# =========================
# PREDICT PAGE (MULTI IMAGE)
# =========================
if page == "🔍 Predict":

    st.subheader("Upload Waste Images")

    uploaded_files = st.file_uploader(
        "Upload images (multiple supported)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.success(f"{len(uploaded_files)} images uploaded")

        cols = st.columns(3)

        for i, file in enumerate(uploaded_files):

            image = Image.open(file).convert("RGB")
            img_array = np.array(image)

            label, confidence = predict(img_array)

            st.session_state.history.append({
                "label": label,
                "confidence": confidence
            })

            with cols[i % 3]:
                st.image(image, use_container_width=True)
                st.success(label)
                st.progress(confidence / 100)
                st.write(f"{confidence:.2f}%")

# =========================
# DASHBOARD
# =========================
elif page == "📊 Dashboard":

    st.subheader("Analytics Dashboard")

    total = st.session_state.organic + st.session_state.recyclable

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Predictions", total)
    col2.metric("Organic Waste", st.session_state.organic)
    col3.metric("Recyclable Waste", st.session_state.recyclable)

    st.markdown("---")

    chart_data = pd.DataFrame({
        "Category": ["Organic", "Recyclable"],
        "Count": [st.session_state.organic, st.session_state.recyclable]
    })

    st.bar_chart(chart_data.set_index("Category"))

    st.markdown("---")

    st.subheader("Recent Predictions")

    if st.session_state.history:
        for item in st.session_state.history[-10:]:
            st.write(f"• {item['label']} ({item['confidence']:.2f}%)")
    else:
        st.info("No predictions yet")

# =========================
# FOOTER
# =========================
st.markdown("""
---
🌍 Environmental Intelligence | 🤖 Deep Learning Powered | 📈 Real-Time Analytics
""")
