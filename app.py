import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AI Skin Disease Diagnosis",
    layout="centered"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIDENCE_THRESHOLD = 0.70  # 🔥 Rejection threshold

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
        torch.load(
            "fine_tuned_skin_disease_model_unfrozen.pth",
            map_location=device
        )
    )
    model.eval()
    return model.to(device)

model = load_model()

# ---------------- IMAGE TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- GRAD-CAM ----------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)


    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[:, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam

def overlay_cam(image, cam):
    image = np.array(image.resize((224, 224)))
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )
    overlay = heatmap * 0.4 + image * 0.6
    return np.uint8(overlay)

# ---------------- UI ----------------
st.markdown(
    """
    <h1 style='text-align: center;'>🧠 AI Skin Disease Diagnosis</h1>
    <p style='text-align: center; color: gray;'>
    Educational & Assistive Tool • Not a Medical Diagnosis
    </p>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "📤 Upload a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("---")
    st.image(image, caption="Uploaded Image", width=350)

    # ---------------- PREDICTION ----------------
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    # ---------------- RESULT ----------------
    st.markdown("---")
    st.markdown("## 🩺 Diagnosis Result")

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("🟡 No disease confidently detected")

        st.write(
            """
            The image does **not strongly match** any of the 8 supported skin diseases.

            **Possible reasons:**
            - Normal / healthy skin
            - Unsupported skin condition
            - Poor lighting or image quality

            ⚠️ Please consult a medical professional for accurate diagnosis.
            """
        )

        st.metric(
            label="Highest Model Confidence",
            value=f"{confidence * 100:.2f}%"
        )
        st.progress(confidence)

        st.info(
            "🧠 **Confidence-based rejection** is used to avoid incorrect predictions "
            "when the model is uncertain."
        )

        st.stop()  # 🔥 Prevent Grad-CAM & false diagnosis

    # ---------------- CONFIDENT PREDICTION ----------------
    st.success(f"**Predicted Condition:** {CLASS_NAMES[pred_idx]}")
    st.metric(
        label="Confidence Score",
        value=f"{confidence * 100:.2f}%"
    )
    st.progress(confidence)

    # ---------------- GRAD-CAM ----------------
    gradcam = GradCAM(model, model.layer4[-1])
    cam = gradcam.generate(input_tensor, pred_idx)
    cam_image = overlay_cam(image, cam)

    st.markdown("---")
    st.markdown("## 🔍 Explainable AI (Grad-CAM)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Image**")
        st.image(image, width=320)

    with col2:
        st.markdown("**Model Attention Heatmap**")
        st.image(cam_image, width=320)

    # ---------------- XAI EXPLANATION ----------------
    st.info(
        """
        🧠 **How to interpret Grad-CAM:**
        - Red / yellow regions show where the model focused most
        - Helps verify attention on clinically relevant skin areas
        - Improves transparency and trust in predictions
        """
    )

    st.markdown(
        "<p style='font-size: 12px; color: gray;'>"
        "⚠️ This system is for educational use only and should not replace professional medical advice."
        "</p>",
        unsafe_allow_html=True
    )
