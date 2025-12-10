import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

class CustomTrOCR:
	@staticmethod
	def from_pretrained(repo_id, device="cpu"):
    		model_path = hf_hub_download(repo_id, filename="weights.pt")  

    		state_dict = torch.load(model_path, map_location=device)

    		processor = TrOCRProcessor.from_pretrained(repo_id)
    		model = VisionEncoderDecoderModel.from_pretrained(repo_id)

    		model.load_state_dict(state_dict, strict=False)
    		model.to(device)
    		model.eval()

    		return model, processor
	
