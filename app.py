import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Skin Disease Diagnosis", layout="centered")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "BA-cellulitis",
    "BA-impetigo",
    "FU-athlete-foot",
    "FU-nail-fungus",
    "FU-ringworm",
    "PA-cutaneous-larva-migrans",
    "VI-chickenpox",
    "VI-shingles"
]

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(
        torch.load("fine_tuned_skin_disease_model_unfrozen.pth", map_location=device)
    )
    model.eval()
    return model.to(device)

model = load_model()

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- UI ----------------
st.title("🧠 AI Skin Disease Diagnosis")
st.markdown("⚠️ *For educational purposes only. Not a medical diagnosis.*")

uploaded_file = st.file_uploader(
    "Upload a skin image", type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=400)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    # ---------------- RESULTS ----------------
    st.success(f"### 🩺 Prediction: {CLASS_NAMES[pred_idx]}")
    st.write(f"**Confidence Score:** {confidence * 100:.2f}%")

    # Confidence bar (nice UI touch)
    st.progress(confidence)
