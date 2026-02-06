import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Siamese CNN Model (Demo)
# ----------------------------
class SignatureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 30 * 30, 128),
            nn.ReLU()
        )

    def forward_once(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ----------------------------
# Load model (demo mode)
# ----------------------------
model = SignatureCNN().to(device)
model.eval()

# ----------------------------
# Image transform
# ----------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ----------------------------
# Extract signature from document
# ----------------------------
def extract_signature_from_document(doc_path):
    img = cv2.imread(doc_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    contours = [c for c in contours if cv2.contourArea(c) > 500]
    if len(contours) == 0:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    signature = img[y:y+h, x:x+w]

    out_path = os.path.join(tempfile.gettempdir(), "extracted_signature.png")
    cv2.imwrite(out_path, signature)
    return out_path

# ----------------------------
# Verify signature
# ----------------------------
def verify_signature(ref_path, test_path, threshold=0.6):
    img1 = Image.open(ref_path).convert("L")
    img2 = Image.open(test_path).convert("L")

    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        f1 = model.forward_once(img1)
        f2 = model.forward_once(img2)
        distance = torch.nn.functional.pairwise_distance(f1, f2)

    return distance.item(), distance.item() < threshold

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Signature Verification", layout="centered")
st.title("✍️ Signature Verification System")

st.markdown("""
Upload:
- 📄 A **document containing a signature**
- ✍️ A **reference (original) signature**

The system will extract the signature from the document and verify it.
""")

doc_file = st.file_uploader(
    "📄 Upload Document (PNG / JPG)", type=["png", "jpg", "jpeg"]
)
ref_file = st.file_uploader(
    "✍️ Upload Reference Signature (PNG / JPG)", type=["png", "jpg", "jpeg"]
)

if doc_file and ref_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as doc_tmp:
        doc_tmp.write(doc_file.read())
        doc_path = doc_tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as ref_tmp:
        ref_tmp.write(ref_file.read())
        ref_path = ref_tmp.name

    st.image(ref_path, caption="Reference Signature", width=300)
    st.image(doc_path, caption="Uploaded Document", width=300)

    if st.button("🔍 Verify Signature"):
        extracted_path = extract_signature_from_document(doc_path)

        if extracted_path is None:
            st.error("❌ Signature could not be detected in the document.")
        else:
            st.image(extracted_path, caption="Extracted Signature", width=300)

            distance, is_genuine = verify_signature(ref_path, extracted_path)

            st.write(f"**Similarity Distance:** `{distance:.4f}`")

            if is_genuine:
                st.success("✅ Signature is GENUINE")
            else:
                st.error("❌ Signature is FORGED")
