# 📝 Handwritten OCR App (TrOCR + Custom Fine-Tuning)

This repository contains a fully-featured handwritten text recognition
system built using **Microsoft TrOCR** and fine-tuned on a custom
dataset of handwritten names. The project includes a **Streamlit UI**, a
**custom model loader**, and the full inference pipeline using a
fine-tuned model hosted on **HuggingFace**.

------------------------------------------------------------------------

## 🚀 Overview

The goal of this project is to accurately recognize handwritten **first
names** and **surnames** using a deep-learning model based on
**TrOCR-base-handwritten (VisionEncoderDecoder)**.\
A custom LoRA-merged model was trained using **100,000 handwritten
samples** extracted from a 400k+ handwriting dataset.

The final model achieved:

-   **WER (Word Error Rate): \~0.81 at epoch 2**
-   **Improved stability over base TrOCR**
-   Significantly enhanced performance on cursive and messy handwriting

------------------------------------------------------------------------

## 🧠 Model Architecture

### ✔ Base Model

-   **microsoft/trocr-base-handwritten**
-   Vision Transformer (ViT) encoder\
-   BART-style text decoder

### ✔ Custom Fine-Tuning

You trained TrOCR using:

-   **LoRA adapters**
-   **100k sample subset**
-   **FP16 training**
-   **Batch size optimized for GPU VRAM**
-   **Checkpoints saved every epoch**

A merged checkpoint (`weights.pt`) is stored on your HuggingFace model
repo:

`scientist-bunny/coustom_ocr_handwritten_model`

------------------------------------------------------------------------

## 📊 Dataset Description

This dataset consists of **400,000+ handwritten names** collected
globally as part of charity projects supporting disadvantaged children.

### ✔ Characteristics

-   206,799 **first names**
-   207,024 **surnames**
-   Natural handwriting from thousands of individuals
-   High variation in style, slant, pressure, spacing
-   Noise, blur, stroke overlap typical in real handwriting

### ✔ Splits

  Split        Size
  ------------ ---------
  Train        331,059
  Test         41,382
  Validation   41,382

➡ **You used 100,000 images** from these splits as your training subset.

------------------------------------------------------------------------

## 📈 Training Performance

### 🟦 Word Error Rate (WER)

  Epoch  ,       WER
  -------------- ------------
  1              \~1.03 \
  2             \~0.81\
  3             \~0.6599\
  **4(Best)**        **\~0.60**

------------------------------------------------------------------------

## 🏗 Repository Structure

    OCR_app/
    │
    ├── model/
    │   ├── model.py                # Custom model loader
    │
    ├── app/
    │   ├── streamlit_app.py        # UI for inference
    │
    ├── requirements.txt
    ├── README.md

------------------------------------------------------------------------

## 🛠 Custom Model Loader Example

``` python
from model.model import CustomTrOCR

model, processor = CustomTrOCR.from_pretrained(
    "scientist-bunny/coustom_ocr_handwritten_model",
    filename="weights.pt",
    device="cuda"
)
```

------------------------------------------------------------------------

## 🖥 Streamlit Inference App

``` bash
streamlit run app/streamlit_app.py
```

------------------------------------------------------------------------

## 📦 Installation

``` bash
git clone https://github.com/bunny-ml/OCR_app
cd OCR_app
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🧪 Example Usage

``` python
from PIL import Image
import torch
from model.model import CustomTrOCR

device = "cuda" if torch.cuda.is_available() else "cpu"

model, processor = CustomTrOCR.from_pretrained(
    "scientist-bunny/coustom_ocr_handwritten_model",
    filename="weights.pt",
    device=device
)

image = Image.open("sample.jpg").convert("RGB")

pixel = processor(image, return_tensors="pt").pixel_values.to(device)
ids = model.generate(pixel, max_length=64)
print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```

------------------------------------------------------------------------

## 🔮 Future Work

-   Train on full 400k dataset\
-   Add paragraph mode\
-   Improve accuracy via augmentation\
-   Convert to ONNX for faster inference

------------------------------------------------------------------------

## 🙏 Acknowledgements

-   Microsoft Research --- TrOCR\
-   HuggingFace Transformers\
-   Appen handwriting dataset\
-   Streamlit
