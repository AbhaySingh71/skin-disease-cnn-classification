import streamlit as st
import numpy as np
from PIL import Image
import keras.utils as image
from keras.models import model_from_json

# -------------------------------
# App Config
# -------------------------------
st.set_page_config(
    page_title="Skin Disease Detection",
    page_icon="🩺",
    layout="centered"
)

st.title("🧬 Skin Disease Detection System")
st.write("Upload a skin image to detect the disease using a deep learning model.")

# -------------------------------
# Classes
# -------------------------------
SKIN_CLASSES = {
    0: 'Actinic Keratoses (Solar Keratoses) or Bowen’s disease',
    1: 'Basal Cell Carcinoma',
    2: 'Benign Keratosis',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic Nevi',
    6: 'Vascular skin lesion'
}

# -------------------------------
# Medicine Recommendation
# -------------------------------
def find_medicine(pred):
    if pred == 0:
        return "Fluorouracil (5-FU)"
    elif pred == 1:
        return "Aldara (Imiquimod)"
    elif pred == 2:
        return "No medicine required"
    elif pred == 3:
        return "Fluorouracil"
    elif pred == 4:
        return "Fluorouracil (5-FU)"
    elif pred == 5:
        return "Fluorouracil"
    elif pred == 6:
        return "Consult Dermatologist"

# -------------------------------
# Load Model (Cached)
# -------------------------------
@st.cache_resource
def load_model():
    with open("model.json", "r") as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights("model.h5")
    return model

model = load_model()

# -------------------------------
# Image Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Skin Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img_pil = Image.open(uploaded_file).convert("RGB")
    st.image(img_pil, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img = img_pil.resize((224, 224))
    img = np.array(img)
    img = img.reshape((1, 224, 224, 3))
    img = img / 255.0

    # Prediction Button
    if st.button("🔍 Detect Disease"):
        with st.spinner("Analyzing image..."):
            prediction = model.predict(img)
            pred = np.argmax(prediction)
            confidence = round(prediction[0][pred] * 100, 2)

        disease = SKIN_CLASSES[pred]
        medicine = find_medicine(pred)

        st.success("✅ Detection Completed")

        st.subheader("🧾 Result")
        st.write(f"**Disease:** {disease}")
        st.write(f"**Confidence:** {confidence}%")
        st.write(f"**Suggested Medicine:** {medicine}")

        if pred == 2:
            st.info("This condition is **benign** and usually does not require treatment.")
        elif pred == 4:
            st.warning("⚠️ Melanoma detected. Please consult a dermatologist immediately.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("⚕️ AI-based Skin Disease Detection | Built with Streamlit")
