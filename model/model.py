import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

class CustomTrOCR:
    @staticmethod
    def from_pretrained(model_path, device="cpu"):
        """
        Loads TrOCR model and custom LoRA/PT weights.
        """

        # load base architecture
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

        # load custom weights
        state_dict = torch.load(f"{model_path}/weights.pt", map_location=device)
        model.load_state_dict(state_dict, strict=False)

        # load processor
        processor = TrOCRProcessor.from_pretrained(model_path)

        return model, processor
