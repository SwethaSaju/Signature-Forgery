import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import io

# ------------------ MODEL ------------------

class SiameseNetwork(nn.Module):
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
            nn.Linear(64 * 29 * 29, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

# ------------------ LOAD MODEL ------------------

@st.cache_resource
def load_model():
    model = SiameseNetwork()

    state = torch.load(
        "siamese_signature.pth",
        map_location="cpu",
        weights_only=False   # 🔥 CRITICAL FIX
    )

    model.load_state_dict(state)
    model.eval()
    return model

# ------------------ IMAGE UTILS ------------------

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess(img):
    return transform(img).unsqueeze(0)

def extract_signature_from_document(img):
    img_gray = img.convert("L")
    img_np = np.array(img_gray)

    thresh = threshold_otsu(img_np)
    binary = img_np < thresh

    labeled = label(binary)
    regions = regionprops(labeled)

    if not regions:
        return None

    largest = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = largest.bbox
    cropped = img_np[minr:maxr, minc:maxc]

    return Image.fromarray(cropped)

def similarity_score(img1, img2):
    with torch.no_grad():
        o1, o2 = model(preprocess(img1), preprocess(img2))
        dist = torch.nn.functional.pairwise_distance(o1, o2)
        return dist.item()

# ------------------ STREAMLIT UI ------------------

st.title("✍️ Signature Forgery Detection")

doc = st.file_uploader("Upload Signed Document", ["png", "jpg", "jpeg"])
ref = st.file_uploader("Upload Reference Signature", ["png", "jpg", "jpeg"])

if doc and ref:
    document_img = Image.open(doc)
    reference_img = Image.open(ref)

    extracted = extract_signature_from_document(document_img)

    if extracted is None:
        st.error("No signature detected in document.")
    else:
        score = similarity_score(extracted, reference_img, model)

        st.image([extracted, reference_img], caption=["Extracted", "Reference"], width=200)

        st.subheader(f"Similarity Distance: {score:.4f}")

        if score < 1.0:
            st.success("✅ Genuine Signature")
        else:
            st.error("❌ Forged Signature")


