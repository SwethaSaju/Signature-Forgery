import os
import cv2
import torch
import gdown
import tempfile
import numpy as np
import streamlit as st
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from pdf2image import convert_from_bytes

# -------------------------------
# CONFIG
# -------------------------------
MODEL_URL = "https://drive.google.com/uc?id=1u7JfKw5dzp8xxkeSlpO8XrGxRNuI7f9m"
MODEL_PATH = "siamese_signature.pth"
IMAGE_SIZE = 128
THRESHOLD = 0.65   # similarity threshold (can tune)

st.set_page_config(page_title="Signature Verification", layout="centered")

# -------------------------------
# MODEL DEFINITION
# -------------------------------
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 30 * 30, 256),
            nn.ReLU()
        )

    def forward_once(self, x):
        x = self.cnn(x)
        return self.fc(x)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

# -------------------------------
# LOAD MODEL (Google Drive)
# -------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(
                "https://drive.google.com/uc?id=1u7JfKw5dzp8xxkeSlpO8XrGxRNuI7f9m",
                MODEL_PATH,
                quiet=False
            )

    model = SiameseNetwork()
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# -------------------------------
# TRANSFORM
# -------------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -------------------------------
# IMAGE UTILS
# -------------------------------
def preprocess_image(img):
    img = img.convert("RGB")
    return transform(img).unsqueeze(0)

def cosine_similarity(a, b):
    return F.cosine_similarity(a, b).item()

# -------------------------------
# SIGNATURE EXTRACTION
# -------------------------------
def extract_signature_from_image(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(img, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    largest = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(largest)
    cropped = img[y:y+h, x:x+w]

    return Image.fromarray(cropped)

def extract_from_pdf(pdf_bytes):
    images = convert_from_bytes(pdf_bytes)
    return extract_signature_from_image(images[0])

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("✍️ Signature Forgery Detection")
st.write("Upload a **reference signature** and a **document (image or PDF)**.")

ref_file = st.file_uploader("Upload Reference Signature", type=["png","jpg","jpeg"])
doc_file = st.file_uploader("Upload Document (Image / PDF)", type=["png","jpg","jpeg","pdf"])

if ref_file and doc_file:
    ref_img = Image.open(ref_file)
    st.image(ref_img, caption="Reference Signature", width=250)

    if doc_file.type == "application/pdf":
        signature_img = extract_from_pdf(doc_file.read())
    else:
        signature_img = extract_signature_from_image(Image.open(doc_file))

    if signature_img is None:
        st.error("❌ Signature not detected in document")
    else:
        st.image(signature_img, caption="Extracted Signature", width=250)

        with torch.no_grad():
            ref_tensor = preprocess_image(ref_img)
            doc_tensor = preprocess_image(signature_img)
            emb1, emb2 = model(ref_tensor, doc_tensor)
            score = cosine_similarity(emb1, emb2)

        st.markdown("---")
        st.write(f"### Similarity Score: **{score:.3f}**")

        if score >= THRESHOLD:
            st.success("✅ Genuine Signature")
        else:
            st.error("❌ Forged Signature")

