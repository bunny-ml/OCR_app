import streamlit as st
import torch
from PIL import Image
from model import CustomTrOCR

st.title("Handwriting Recognition (TrOCR + LoRA)")

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = CustomTrOCR.from_pretrained(
        "your-username/trocr-lora-handwriting", 
        device=device
    )
    model.to(device)
    return model, processor, device

model, processor, device = load_model()

img_file = st.file_uploader("Upload handwriting", type=["png", "jpg", "jpeg"])

if img_file:
    image = Image.open(img_file).convert("RGB")
    st.image(image, width=300)

    pixel = processor(image, return_tensors="pt").pixel_values.to(device)
    ids = model.generate(pixel, max_length=64)

    text = processor.batch_decode(ids, skip_special_tokens=True)[0]

    st.subheader("Recognized text:")
    st.write(text)
