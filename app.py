import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Solar Panel Defect Detection",
    page_icon="☀️",
    layout="centered"
)

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("trained_effnet_finetune.h5")
    return model

model = load_model()

# -------------------------
# CLASS NAMES (EDIT IF NEEDED)
# -------------------------
class_names = [
    "Clean",
    "Dusty",
    "Bird Drop",
    "Electrical Damage",
    "Physical Damage",
    "Snow Covered"
]

# -------------------------
# PREPROCESS FUNCTION
# -------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image)

    img_array = preprocess_input(img_array)  # EfficientNet preprocessing
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# -------------------------
# UI
# -------------------------
st.title("☀️ Solar Panel Defect Detection")
st.write("Upload a solar panel image to detect defects.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

# -------------------------
# PREDICTION
# -------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # ✅ Updated (no deprecated param)
    st.image(image, caption="Uploaded Image", width="stretch")

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing panel..."):
            processed = preprocess_image(image)

            preds = model.predict(processed)
            predicted_class = class_names[np.argmax(preds)]
            confidence = np.max(preds)

        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2%}")

        # -------------------------
        # PROBABILITY BREAKDOWN
        # -------------------------
        st.subheader("📊 Class Probabilities")

        prob_dict = {
            class_names[i]: float(preds[0][i])
            for i in range(len(class_names))
        }

        st.bar_chart(prob_dict)