import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import pdfplumber

st.set_page_config(page_title="Signature Forgery Detection", layout="centered")
st.title("✍️ Signature Forgery Detection")
st.write("Upload **two PDF files** containing signatures to verify authenticity.")

# --- Helper Functions ---
def pdf_to_image(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        page = pdf.pages[0]
        if page.images:
            img_obj = page.images[0]
            bbox = (img_obj["x0"], img_obj["top"], img_obj["x1"], img_obj["bottom"])
            pil_image = page.to_image(resolution=200).original.crop(bbox)
            return np.array(pil_image)
        else:
            st.error("No images found in PDF.")
            return None

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

    if img1 is not None and img2 is not None:
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
