import soundfile as sf
import os
import torch
from pathlib import Path
from voxcpm import VoxCPM

class VoxCPMEngine:
    def __init__(self, model_path: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

        if model_path:
            self.model_path = Path(model_path)
        else:
            current_dir = Path(__file__).parent.parent
            self.model_path = current_dir / "model_weights" / "VoxCPM2"

        print(f"📁 模型路徑: {self.model_path}")

    def load_model(self):
        self.model = VoxCPM.from_pretrained(
            "openbmb/VoxCPM2",
            load_denoiser=False,
            cache_dir="./model_weights"
        )
        self.model.to(self.device)

    def generate(self, **kwargs):
        if self.model is None:
            raise RuntimeError("模型尚未載入")
        wav = self.model.generate(**kwargs)
        return wav, self.model.tts_model.sample_rate
