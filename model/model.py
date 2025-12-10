import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from huggingface_hub import hf_hub_download

class CustomTrOCR:
    @staticmethod
    def from_pretrained(repo_id, filename="weights.pt", device="cpu"):
        
        # Download weights.pt (or mods.pt)
        model_path = hf_hub_download(repo_id, filename=filename)

        # Load state dict
        state_dict = torch.load(model_path, map_location=device)

        # Load processor + base model from repo
        processor = TrOCRProcessor.from_pretrained(repo_id)
        model = VisionEncoderDecoderModel.from_pretrained(repo_id)

        # Load your fine-tuned LoRA weights
        model.load_state_dict(state_dict, strict=False)

        model.to(device)
        model.eval()
        return model, processor

