import os
import torch
import torchaudio
from PIL import Image
from typing import List, Tuple

def scan_xcapture(root_dir: str) -> List[Tuple[str, str]]:
    samples = []
    for scene_id in sorted(os.listdir(root_dir)):
        scene_path = os.path.join(root_dir, scene_id)
        if not os.path.isdir(scene_path):
            continue
        for clip_id in sorted(os.listdir(scene_path)):
            clip_path = os.path.join(scene_path, clip_id)
            if not os.path.isdir(clip_path):
                continue
            rgb_path = os.path.join(clip_path, "vision", "rgb.png")
            audio_path = os.path.join(clip_path, "audio", "audio.wav")
            if os.path.exists(rgb_path) and os.path.exists(audio_path):
                samples.append((rgb_path, audio_path))
    return samples

class XCaptureDataset(torch.utils.data.Dataset):
    def __init__(self, samples, csv_path, target_sr=16000):
        self.samples = samples
        self.target_sr = target_sr
        df = pd.read_csv(csv_path)
        self.txt_mapping = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_p, wav_p = self.samples[idx]
        text = self.txt_mapping.get(rgb_p, "a photo of ")
        img = Image.open(rgb_p).convert("RGB")
        wav, sr = torchaudio.load(wav_p)
        wav = wav.mean(dim=0)
        if sr != self.target_sr: 
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        return {"image": img, "audio": wav.numpy().astype("float32"), "text": text}