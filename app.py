import os
import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from pdf2image import convert_from_bytes
import urllib.request

# ---------------- CONFIG ----------------
MODEL_URL = "https://huggingface.co/SwethSwetha/signature-forgery-model/resolve/main/siamese_signature.pth"
MODEL_PATH = "siamese_signature.pth"
IMAGE_SIZE = 128
THRESHOLD = 0.65

st.set_page_config(page_title="Signature Forgery Detection", layout="centered")

# ---------------- MODEL ----------------
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
        return self.fc(self.cnn(x))

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    model = SiameseNetwork()

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False   # ✅ REQUIRED for PyTorch 2.6+
    )

    model.load_state_dict(state_dict)
    model.eval()
    return model

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess(img):
    return transform(img).unsqueeze(0)

# ---------------- SIGNATURE EXTRACTION ----------------
def extract_signature_from_image(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cropped = gray[y:y+h, x:x+w]
    return Image.fromarray(cropped)

def extract_signature_from_pdf(pdf_bytes):
    pages = convert_from_bytes(pdf_bytes)
    return extract_signature_from_image(pages[0]) if pages else None

# ---------------- UI ----------------
st.title("✍️ Signature Forgery Detection")
st.write("Upload a **reference signature** and a **document (image or PDF)**.")

ref_file = st.file_uploader("Reference Signature", type=["png", "jpg", "jpeg"])
doc_file = st.file_uploader("Document (Image or PDF)", type=["png", "jpg", "jpeg", "pdf"])

if ref_file and doc_file:
    ref_img = Image.open(ref_file)
    st.image(ref_img, caption="Reference Signature", width=250)

    if doc_file.type == "application/pdf":
        sig_img = extract_signature_from_pdf(doc_file.read())
    else:
        sig_img = extract_signature_from_image(Image.open(doc_file))

    if sig_img is None:
        st.error("Signature not detected. Please upload a clearer document.")
    else:
        st.image(sig_img, caption="Extracted Signature", width=250)

        with torch.no_grad():
            e1, e2 = model(preprocess(ref_img), preprocess(sig_img))
            score = F.cosine_similarity(e1, e2).item()

        st.markdown("---")
        st.write(f"### Similarity Score: **{score:.3f}**")

        if score >= THRESHOLD:
            st.success("✅ GENUINE SIGNATURE")
        else:
            st.error("❌ FORGED SIGNATURE")
