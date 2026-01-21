import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor, CLIPProcessor
import torchaudio
from PIL import Image
from tqdm import tqdm
import wandb
import random
from typing import List, Tuple

# 导入
from models import UnifiedAligner
from losses import DualAnchorContrastiveLoss
from evaluator import DistributedMultiModalEvaluator

class Config:
    audio_encoder_path = "/data1/yizhuo/ULIA/ckpt/models--Atotti--Qwen3-Omni-AudioTransformer/snapshots/cf943f056d4bd5f647a735a02efae7aa2d772af1"
    qwen_processor_path = "/data1/yizhuo/ULIA/ckpt/Qwen3-Omni-30B-A3B-Instruct"
    clip_path = "/data1/yizhuo/ULIA/ckpt/CLIP-B-16"
    data_root = "/data1/yizhuo/XCapture_data"
    
    train_ratio = 0.9
    target_sr = 16000
    batch_size = 32  # 每个 GPU 的 batch_size
    epochs = 150
    lr = 1e-4
    use_fp16 = True
    use_flash_attention = False

# --- Data Helpers ---
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
    def __init__(self, samples, target_sr=16000):
        self.samples = samples
        self.target_sr = target_sr

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_p, wav_p = self.samples[idx]
        img = Image.open(rgb_p).convert("RGB")
        wav, sr = torchaudio.load(wav_p)
        wav = wav.mean(dim=0)
        if sr != self.target_sr: 
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        return {"image": img, "audio": wav.numpy().astype("float32")}

def get_collate_fn(clip_p, audio_p):
    def collate(batch):
        img_in = clip_p(images=[b["image"] for b in batch], return_tensors="pt")
        aud_in = audio_p([b["audio"] for b in batch], sampling_rate=16000, padding=True, return_tensors="pt")
        return {"pixel_values": img_in["pixel_values"], "audio_values": aud_in["input_features"], "audio_attention_mask": aud_in.get("attention_mask")}
    return collate

# --- Main Training ---
def main():
    # 1. Initialize DDP
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    cfg = Config()

    # 2. Model & Loss (Wrapped in DDP)
    model = UnifiedAligner(cfg).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    criterion = DualAnchorContrastiveLoss().to(device)
    evaluator = DistributedMultiModalEvaluator(device=device, ks=[1, 10, 30])

    # 3. Data
    samples = scan_xcapture(cfg.data_root)
    random.shuffle(samples)
    split = int(len(samples) * cfg.train_ratio)

    train_samples = samples[:split]
    val_samples = samples[split:]
    dataset = XCaptureDataset(train_samples, cfg.target_sr)
    val_dataset = XCaptureDataset(val_samples, cfg.target_sr)

    clip_proc = CLIPProcessor.from_pretrained(cfg.clip_path)
    audio_proc = AutoProcessor.from_pretrained(cfg.qwen_processor_path).feature_extractor
    
    sampler = DistributedSampler(dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, 
                        collate_fn=get_collate_fn(clip_proc, audio_proc), num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, 
                        collate_fn=get_collate_fn(clip_proc, audio_proc), num_workers=4, pin_memory=True)
    if rank == 0:
        print(f"train_loader {len(train_samples)} val_loader {len(val_samples)}")
    # 4. Optimizer & Scaler
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_fp16)

    if cfg.use_fp16 and device.type == "cuda":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.float32

    if rank == 0:
        wandb.init(project="MultiModal-Align", name="Unified-Run")

    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)
        model.train()
        model.module.clip.eval()
        pbar = tqdm(loader, disable=(rank != 0))

        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=(device.type == "cuda")):
                outputs = model(batch)
                loss_dict = criterion(outputs)
                loss = loss_dict['loss/loss']

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if rank == 0:
                wandb.log(loss_dict)
                pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")

        val_metrics = evaluator.evaluate(model, val_loader)
            
        if rank == 0:
            print(f"Epoch {epoch} Metrics: {val_metrics}")
            wandb.log(val_metrics)
        
        dist.barrier()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()