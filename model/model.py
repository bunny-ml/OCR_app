import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from huggingface_hub import hf_hub_download

class CustomTrOCR:
    @staticmethod
    def from_pretrained(repo_id, filename="weights.pt", device="cpu"):
        # download the file (weights.pt / mods.pt)
        weights_path = hf_hub_download(repo_id, filename=filename)

        # load the weights
        state_dict = torch.load(weights_path, map_location=device)

        # load model + processor from repo
        processor = TrOCRProcessor.from_pretrained(repo_id)
        model = VisionEncoderDecoderModel.from_pretrained(repo_id)

        # apply your trained weights
        model.load_state_dict(state_dict, strict=False)

        model.to(device)
        model.eval()
        return model, processor

