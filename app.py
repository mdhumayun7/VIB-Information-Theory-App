import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from vib_model import VIB

st.set_page_config(page_title="VIB Research Demo")

# Load model safely
model = VIB()
model.load_state_dict(torch.load("vib_model.pth", map_location="cpu"))
model.eval()

st.title("Deep Variational Information Bottleneck")

uploaded = st.file_uploader("Upload Digit Image", type=["png","jpg","jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("L")

    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1-x),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits, mu, logvar, z = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

        kl = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        ).item()

    st.image(image, width=150)
    st.success(f"Prediction: {pred.item()}")
    st.write(f"Confidence: {torch.max(probs).item():.4f}")
    st.write(f"I(Z;X) ≈ {kl:.4f}")
    st.write(f"I(Z;Y) ≈ {-F.cross_entropy(logits, pred).item():.4f}")

    # Latent 2D
    if z.shape[1] >= 2:
        fig, ax = plt.subplots()
        ax.scatter(z[0,0].item(), z[0,1].item())
        ax.set_title("Latent Representation (2D projection)")
        st.pyplot(fig)
