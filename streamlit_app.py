import streamlit as st
import torch
from PIL import Image
from model.model import CustomTrOCR
from transformers import AutoConfig, AutoProcessor

st.title("Handwriting Recognition (TrOCR + LoRA)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    repo_id = "scientist-bunny/coustom_ocr_handwritten_model"

    # Load config + processor
    config = AutoConfig.from_pretrained(repo_id)
    processor = AutoProcessor.from_pretrained(repo_id)

    # Initialize the model
    model = CustomTrOCR(config)

    # Load weights from HF (.pt file)
    weights = torch.hub.load_state_dict_from_url(
        f"https://huggingface.co/{repo_id}/resolve/main/weights.pt",
        map_location=device
    )
    model.load_state_dict(weights)

    model.to(device)
    model.eval()

    return model, processor


model, processor = load_model()


# UI
img_file = st.file_uploader("Upload handwriting image", type=["png", "jpg", "jpeg"])

if img_file:
    image = Image.open(img_file).convert("RGB")
    st.image(image, width=300)

    # Preprocess
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    # Generate output text
    ids = model.generate(pixel_values, max_length=64)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]

    st.subheader("Recognized text:")
    st.write(text)

