import os
import cv2
import torch
import gdown
import numpy as np
import streamlit as st
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from pdf2image import convert_from_bytes

# ---------------- CONFIG ----------------
MODEL_URL = "https://drive.google.com/uc?id=1u7JfKw5dzp8xxkeSlpO8XrGxRNuI7f9m"
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
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = SiameseNetwork()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess(img):
    return transform(img).unsqueeze(0)

def cosine_sim(a, b):
    return F.cosine_similarity(a, b).item()

# ---------------- SIGNATURE EXTRACTION ----------------
def extract_signature(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    x,y,w,h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cropped = gray[y:y+h, x:x+w]
    return Image.fromarray(cropped)

def extract_from_pdf(pdf_bytes):
    pages = convert_from_bytes(pdf_bytes)
    return extract_signature(pages[0])

# ---------------- UI ----------------
st.title("✍️ Signature Forgery Detection")
st.write("Upload a **reference signature** and a **document (image or PDF)**.")

ref_file = st.file_uploader("Reference Signature", type=["png","jpg","jpeg"])
doc_file = st.file_uploader("Document (Image / PDF)", type=["png","jpg","jpeg","pdf"])

if ref_file and doc_file:
    ref_img = Image.open(ref_file)
    st.image(ref_img, caption="Reference Signature", width=250)

    if doc_file.type == "application/pdf":
        sig_img = extract_from_pdf(doc_file.read())
    else:
        sig_img = extract_signature(Image.open(doc_file))

    if sig_img is None:
        st.error("❌ Signature not detected in document")
    else:
        st.image(sig_img, caption="Extracted Signature", width=250)

        with torch.no_grad():
            e1, e2 = model(preprocess(ref_img), preprocess(sig_img))
            score = cosine_sim(e1, e2)

        st.markdown("---")
        st.write(f"### Similarity Score: **{score:.3f}**")

        if score >= THRESHOLD:
            st.success("✅ Genuine Signature")
        else:
            st.error("❌ Forged Signature")
