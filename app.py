import streamlit as st
import cv2
import numpy as np
from pdf2image import convert_from_bytes
from skimage.metrics import structural_similarity as ssim
from PIL import Image

st.set_page_config(page_title="Signature Forgery Detection", layout="centered")

st.title("✍️ Signature Forgery Detection")
st.write("Upload **two PDF files** containing signatures to verify authenticity.")

# --- Helper Functions ---

def pdf_to_image(pdf_file):
    images = convert_from_bytes(pdf_file.read())
    img = images[0]
    img = np.array(img)
    return img

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (300, 150))
    return gray

def calculate_similarity(img1, img2):
    score, _ = ssim(img1, img2, full=True)
    return score

# --- File Upload ---
pdf1 = st.file_uploader("Upload Reference Signature PDF", type=["pdf"])
pdf2 = st.file_uploader("Upload Test Signature PDF", type=["pdf"])

if pdf1 and pdf2:
    img1 = pdf_to_image(pdf1)
    img2 = pdf_to_image(pdf2)

    img1_p = preprocess(img1)
    img2_p = preprocess(img2)

    st.subheader("Extracted Signatures")
    col1, col2 = st.columns(2)
    col1.image(img1, caption="Reference Signature", use_column_width=True)
    col2.image(img2, caption="Test Signature", use_column_width=True)

    similarity = calculate_similarity(img1_p, img2_p)

    st.subheader("🔍 Verification Result")
    st.write(f"**Similarity Score:** `{similarity:.2f}`")

    if similarity > 0.75:
        st.success("✅ Signatures are GENUINE")
    else:
        st.error("❌ Signatures are FORGED")
