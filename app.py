import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from vib_model import VIB

# -----------------------------
# Load Model
# -----------------------------
model = VIB()
model.load_state_dict(torch.load("vib_model.pth", map_location="cpu"))
model.eval()

# -----------------------------
# Load Information Plane Data
# -----------------------------
data = np.load("info_plane_data.npz", allow_pickle=True)
I_ZX_list = data["I_ZX_list"]
I_ZY_list = data["I_ZY_list"]

# -----------------------------
# UI
# -----------------------------
st.title("Information Bottleneck Digit Classifier")

st.write("""
This model implements the Deep Variational Information Bottleneck framework.
It balances compression (I(Z;X)) and prediction (I(Z;Y)).
""")

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader("Upload 28x28 grayscale digit image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")

    st.image(image, caption="Uploaded Image", width=150)

    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        confidence = torch.max(probs).item()

    st.success(f"Predicted Digit: {pred.item()}")
    st.write(f"Confidence Score: {confidence:.4f}")

# -----------------------------
# Information Plane Plot
# -----------------------------
st.subheader("Information Plane")

fig, ax = plt.subplots(figsize=(6,5))

epochs = len(I_ZX_list)
colors = plt.cm.viridis(np.linspace(0,1,epochs))

for i in range(epochs):
    ax.scatter(I_ZX_list[i], I_ZY_list[i], color=colors[i], s=80)
    ax.text(I_ZX_list[i], I_ZY_list[i], f"E{i+1}", fontsize=8)

# Highlight compression phase
compression_start = None
for i in range(1, epochs):
    if I_ZX_list[i] < I_ZX_list[i-1]:
        compression_start = i
        break

if compression_start is not None:
    ax.axvspan(I_ZX_list[compression_start],
               max(I_ZX_list),
               color='red', alpha=0.1)

ax.set_xlabel("I(Z;X) (Compression)")
ax.set_ylabel("I(Z;Y) (Predictive Information)")
ax.set_title("Information Plane Dynamics")
ax.grid(True)

st.pyplot(fig)

# -----------------------------
# Save Figure Button
# -----------------------------
if st.button("Save Information Plane as PNG"):
    fig.savefig("Information_Plane.png", dpi=300)
    st.success("Saved as Information_Plane.png")

