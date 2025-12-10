import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from huggingface_hub import hf_hub_download

class CustomTrOCR:
    @staticmethod
    def from_pretrained(repo_id, filename="weights.pt", device="cpu"):
        # download weights file from hub
        weights_path = hf_hub_download(repo_id, filename=filename)

        # load your trained weights
        state_dict = torch.load(weights_path, map_location=device)

        # load base model + processor
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

        # apply your weights (LoRA merged weights)
        model.load_state_dict(state_dict, strict=False)

        model.to(device)
        model.eval()
        return model, processor

