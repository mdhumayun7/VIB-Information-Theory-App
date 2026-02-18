import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from vib_model import VIB
import os

# --------------------------
# Page Config
# --------------------------
st.set_page_config(page_title="Deep VIB Research Demo")
st.title("Deep Variational Information Bottleneck")

device = torch.device("cpu")

# --------------------------
# Load Model Safely
# --------------------------
model = VIB()  # MUST match training architecture

if os.path.exists("vib_model.pth"):
    model.load_state_dict(torch.load("vib_model.pth", map_location=device))
    model.to(device)
    model.eval()
else:
    st.error("vib_model.pth not found!")
    st.stop()

# --------------------------
# Upload Image
# --------------------------
uploaded = st.file_uploader("Upload MNIST-style Digit", type=["png","jpg","jpeg"])

if uploaded:

    image = Image.open(uploaded).convert("L")

    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1 - x),  # invert to match MNIST
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    # --------------------------
    # Forward Pass
    # --------------------------
    with torch.no_grad():
        logits, mu, logvar, z = model(input_tensor)

        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

        # I(Z;X) via KL
        kl = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp()
        ).item()

        # I(Z;Y) approx via negative CE
        ce = F.cross_entropy(logits, pred)
        I_ZY = -ce.item()

    # --------------------------
    # Display Results
    # --------------------------
    st.image(image, width=150)
    st.success(f"Predicted Digit: {pred.item()}")
    st.write(f"Confidence: {torch.max(probs).item():.4f}")
    st.write(f"I(Z;X) (Compression) ≈ {kl:.4f}")
    st.write(f"I(Z;Y) (Predictive Info) ≈ {I_ZY:.4f}")

    # --------------------------
    # Latent 2D Projection
    # --------------------------
    if z.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(4,4))
        ax.scatter(z[0,0].item(), z[0,1].item(), s=120)
        ax.set_xlabel("Z1")
        ax.set_ylabel("Z2")
        ax.set_title("Latent Representation (2D)")
        ax.grid(True)
        st.pyplot(fig)
