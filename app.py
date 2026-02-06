import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Signature Forgery Detection")
st.title("✍️ Signature Forgery Detection")

# -----------------------------
# Model Definition
# -----------------------------
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 61 * 61, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward_once(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

# -----------------------------
# Image Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def preprocess(img):
    return transform(img).unsqueeze(0)

# -----------------------------
# Load Model SAFELY
# -----------------------------
@st.cache_resource
def load_model():
    checkpoint = torch.load(
        "siamese_signature.pth",
        map_location="cpu",
        weights_only=False  # 🔥 CRITICAL FIX
    )

    # Case 1: full model saved
    if isinstance(checkpoint, nn.Module):
        model = checkpoint

    # Case 2: state_dict saved
    else:
        model = SiameseNetwork()
        model.load_state_dict(checkpoint)

    model.eval()
    return model

model = load_model()

# -----------------------------
# Similarity Function
# -----------------------------
def similarity_score(img1, img2, model):
    with torch.no_grad():
        o1, o2 = model(preprocess(img1), preprocess(img2))
        return F.cosine_similarity(o1, o2).item()

# -----------------------------
# UI
# -----------------------------
ref_file = st.file_uploader("Upload Genuine Signature", type=["png", "jpg", "jpeg"])
test_file = st.file_uploader("Upload Signature to Verify", type=["png", "jpg", "jpeg"])

if ref_file and test_file:
    ref_img = Image.open(ref_file).convert("RGB")
    test_img = Image.open(test_file).convert("RGB")

    st.image([ref_img, test_img], caption=["Reference", "Test"], width=250)

    if st.button("Verify Signature"):
        score = similarity_score(ref_img, test_img, model)

        st.subheader("Similarity Score")
        st.write(f"{score:.4f}")

        if score > 0.80:
            st.success("✅ Genuine Signature")
        else:
            st.error("❌ Forged Signature")

