import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
import base64


st.set_page_config(page_title="AI Detector", layout="centered")


def get_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

BASE_DIR = Path(__file__).resolve().parent.parent
bg_path = BASE_DIR / "deployment/assets/bg.jpeg"
bg_image = get_base64(bg_path)


st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/jpeg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background: radial-gradient(circle at 50% 30%, rgba(79,70,229,0.25), rgba(2,6,23,0.95));
    backdrop-filter: blur(8px);
    z-index: -1;
}}

.block-container {{
    padding-top: 5rem;
    max-width: 850px;
}}

.hero {{
    text-align: center;
    font-size: 52px;
    font-weight: 900;
    letter-spacing: -1px;
    color: #e2e8f0;
    text-shadow: 
        0 0 20px rgba(99,102,241,0.5),
        0 0 40px rgba(99,102,241,0.3);
}}

.sub {{
    text-align: center;
    color: #94a3b8;
    margin-top: 10px;
    margin-bottom: 50px;
}}

.stFileUploader {{
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 22px;
    border-radius: 18px;
    transition: all 0.35s ease;
}}

.stFileUploader:hover {{
    border: 1px solid rgba(99,102,241,0.6);
    box-shadow: 
        0 0 30px rgba(99,102,241,0.25),
        0 0 80px rgba(99,102,241,0.15);
}}

.result-card {{
    margin-top: 40px;
    padding: 30px;
    border-radius: 20px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.08);
    text-align: center;
}}

.real {{
    color: #22c55e;
    font-size: 28px;
    font-weight: 700;
}}

.ai {{
    color: #ef4444;
    font-size: 28px;
    font-weight: 700;
}}

h3 {{
    color: #e2e8f0 !important;
}}

.stProgress > div > div > div > div {{
    background-image: linear-gradient(to right, #6366f1, #a855f7);
}}
</style>
""", unsafe_allow_html=True)


MODEL_PATH = BASE_DIR / "models/saved_models/model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def patch_classification(img, grid_size=6):
    img_resized = img.resize((224, 224))
    img_np = np.array(img_resized)

    patch_size = 224 // grid_size
    results = []

    for i in range(grid_size):
        for j in range(grid_size):
            patch = img_np[
                i*patch_size:(i+1)*patch_size,
                j*patch_size:(j+1)*patch_size
            ]

            patch_img = Image.fromarray(patch)
            patch_tensor = transform(patch_img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(patch_tensor)
                probs = torch.softmax(output, dim=1)
                confidence, pred = torch.max(probs, 1)

            results.append({
                "pos": (i, j),
                "label": pred.item(),
                "confidence": confidence.item()
            })

    return results, img_resized


def create_overlay(img, results, grid_size=6):
    overlay = np.array(img).copy()
    h, w, _ = overlay.shape

    patch_h = h // grid_size
    patch_w = w // grid_size

    for r in results:
        i, j = r["pos"]

        if r["label"] == 0 and r["confidence"] > 0.55:
            y1 = i * patch_h
            y2 = (i + 1) * patch_h
            x1 = j * patch_w
            x2 = (j + 1) * patch_w

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 80, 80), 1)

    return overlay


def generate_explanation(results, final_label):
    total = len(results)

    ai_patches = [r for r in results if r["label"] == 0]
    ai_count = len(ai_patches)

    confidences = [r["confidence"] for r in results]
    variance = np.var(confidences)

    ratio = ai_count / total

    center_hits = sum(1 for r in ai_patches if 1 <= r["pos"][0] <= 4 and 1 <= r["pos"][1] <= 4)
    edge_hits = ai_count - center_hits

    if final_label == "AI Generated":
        if ratio > 0.5:
            return f"Widespread synthetic patterns detected ({ai_count}/{total} regions)."
        elif center_hits > edge_hits:
            return "Synthetic artifacts concentrated near subject."
        else:
            return "Artifacts mostly in background areas."

    else:
        if ratio < 0.15:
            return f"No strong synthetic patterns. Stable prediction (variance {variance:.4f})."
        elif variance > 0.02:
            return "Minor inconsistencies, likely natural variations."
        else:
            return f"Few anomalous regions ({ai_count}/{total}), but overall appears real."


st.markdown('<div class="hero">⚡ AI vs REAL DETECTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Upload an image. We analyze reality.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", label_visibility="collapsed")

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, use_container_width=True)

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    classes = ["AI Generated", "Real"]
    result = classes[pred.item()]
    confidence = confidence.item() * 100

    results, img_resized = patch_classification(img)
    overlay = create_overlay(img_resized, results)

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    st.image(overlay, caption="Suspicious Regions Highlighted", use_container_width=True)

    if result == "Real":
        st.markdown(f'<div class="real">REAL ({confidence:.2f}%)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ai">AI GENERATED ({confidence:.2f}%)</div>', unsafe_allow_html=True)

    st.progress(int(confidence))

    st.markdown("### 🧠 AI Analysis")
    st.markdown(f"<div style='color:#cbd5f5'>{generate_explanation(results, result)}</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)