from voxcpm import VoxCPM
import soundfile as sf
import os
import torch

class VoxCPMEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def load_model(self):
        self.model = VoxCPM.from_pretrained(
        "openbmb/VoxCPM2",
        load_denoiser=False,
        )

    def generate(self, **kwargs):
        if self.model is None:
            raise RuntimeError("模型尚未載入")
        wav = self.model.generate(**kwargs)
        return wav, self.model.tts_model.sample_rate
