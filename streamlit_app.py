import streamlit as st
import torch
from PIL import Image
from model.model import CustomTrOCR

st.title("Handwriting Recognition (TrOCR + LoRA)")

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    repo_id = "scientist-bunny/coustom_ocr_handwritten_model"
    model, processor = CustomTrOCR.from_pretrained(
        repo_id,
        filename="weights.pt",
        device=device
    )
    return model, processor, device


model, processor, device = load_model()


# UI
img_file = st.file_uploader("Upload handwriting image", type=["png", "jpg", "jpeg"])

if img_file:
    image = Image.open(img_file).convert("RGB")
    st.image(image, width=300)

    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    ids = model.generate(pixel_values, max_length=64)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]

    st.subheader("Recognized text:")
    st.write(text)

