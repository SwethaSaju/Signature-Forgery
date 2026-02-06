import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from pdf2image import convert_from_bytes
import tempfile
import os

# -------------------------
# MODEL (same as notebook)
# -------------------------
class SiameseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 30 * 30, 128)
        )

    def forward_once(self, x):
        return self.cnn(x)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    model = SiameseCNN()
    model.load_state_dict(torch.load("siamese_signature.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -------------------------
# TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# -------------------------
# SIGNATURE EXTRACTION
# -------------------------
def extract_signature(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    c = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    return img[y:y+h, x:x+w]

# -------------------------
# PDF → IMAGE
# -------------------------
def pdf_to_image(pdf_bytes):
    images = convert_from_bytes(pdf_bytes)
    return np.array(images[0])

# -------------------------
# VERIFY FUNCTION
# -------------------------
def verify(sig1, sig2):
    sig1 = transform(sig1).unsqueeze(0)
    sig2 = transform(sig2).unsqueeze(0)

    with torch.no_grad():
        f1, f2 = model(sig1, sig2)
        distance = torch.nn.functional.pairwise_distance(f1, f2)

    return distance.item()

# -------------------------
# STREAMLIT UI
# -------------------------
st.title("✍️ Signature Forgery Detection")
st.markdown("Upload **Reference Signature** and **Document / Signature to Verify**")

ref_file = st.file_uploader("Upload Reference Signature", type=["png","jpg","jpeg"])
doc_file = st.file_uploader("Upload Document or Signature", type=["png","jpg","jpeg","pdf"])

threshold = st.slider("Forgery Threshold (lower = stricter)", 0.3, 2.0, 1.0)

if ref_file and doc_file:
    ref_img = Image.open(ref_file).convert("RGB")

    if doc_file.type == "application/pdf":
        doc_img = pdf_to_image(doc_file.read())
    else:
        doc_img = np.array(Image.open(doc_file).convert("RGB"))

    extracted = extract_signature(doc_img)

    if extracted is None:
        st.error("❌ No signature detected in document")
    else:
        test_img = Image.fromarray(extracted)

        dist = verify(ref_img, test_img)

        st.image([ref_img, test_img], caption=["Reference", "Extracted"], width=250)

        st.write(f"### Distance: `{dist:.4f}`")

        if dist < threshold:
            st.success("✅ Genuine Signature")
        else:
            st.error("❌ Forged Signature")
