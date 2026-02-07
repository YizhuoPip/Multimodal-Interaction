import os
import io
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
import pandas as pd
from typing import List, Tuple

from models import UnifiedAligner
from losses import DualAnchorContrastiveLoss
from evaluator import DistributedMultiModalEvaluator
from datasets import scan_xcapture, XCaptureDataset

class Config:
    audio_encoder_path = "/data1/yizhuo/ULIA/ckpt/models--Atotti--Qwen3-Omni-AudioTransformer/snapshots/cf943f056d4bd5f647a735a02efae7aa2d772af1"
    qwen_processor_path = "/data1/yizhuo/ULIA/ckpt/Qwen3-Omni-30B-A3B-Instruct"
    clip_path = "/data1/yizhuo/ULIA/ckpt/CLIP-B-16"
    data_root = "/data1/yizhuo/XCapture_data"
    
    train_ratio = 0.9
    warmup_ratio = 0.1
    target_sr = 16000
    batch_size = 32  # 每个 GPU 的 batch_size
    epochs = 100
    lr = 1e-4
    use_fp16 = True
    use_flash_attention = False
    seed = 42
    use_vision = True
    use_text = True
    use_audio = True

def broadcast_object(obj, src=0):
    """将Python对象从src rank广播到所有rank"""
    rank = dist.get_rank()
    
    if rank == src:
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = buffer.getvalue()
        size = torch.tensor([len(data)], dtype=torch.long, device='cuda')
        data_tensor = torch.ByteTensor(list(data)).cuda()  # 修复这里
    else:
        size = torch.tensor([0], dtype=torch.long, device='cuda')
    
    # 广播大小
    dist.broadcast(size, src=src)
    
    if rank != src:
        data_tensor = torch.empty(size.item(), dtype=torch.uint8, device='cuda')
    
    # 广播数据
    dist.broadcast(data_tensor, src=src)
    
    if rank != src:
        buffer = io.BytesIO(data_tensor.cpu().numpy())
        obj = torch.load(buffer, map_location='cpu')
    
    return obj

def set_seed_for_distributed(seed, rank):
    """为分布式训练设置种子"""
    # 设置随机种子
    random.seed(seed + rank * 1000)
    torch.manual_seed(seed + rank * 1000)
    torch.cuda.manual_seed(seed + rank * 1000)
    torch.cuda.manual_seed_all(seed + rank * 1000)
    
    # 确保确定性计算（防止CUDA随机性）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def get_collate_fn(clip_p, audio_p):
    def collate(batch):
        text_in = clip_p(text=[b["text"] for b in batch], return_tensors="pt", padding=True, truncation=True)
        img_in = clip_p(images=[b["image"] for b in batch], return_tensors="pt")
        aud_in = audio_p([b["audio"] for b in batch], sampling_rate=16000, padding=True, return_tensors="pt")
        return {"input_ids":text_in["input_ids"], "text_attention_mask": text_in.get("attention_mask"), "pixel_values": img_in["pixel_values"], "audio_values": aud_in["input_features"], "audio_attention_mask": aud_in.get("attention_mask")}
        #return {"pixel_values": img_in["pixel_values"], "audio_values": aud_in["input_features"], "audio_attention_mask": aud_in.get("attention_mask")}
    return collate

# --- Main Training ---
def main():
    cfg = Config()
    # Initialize DDP
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # seed
    set_seed_for_distributed(cfg.seed, rank)
    
    if rank == 0:
        samples = scan_xcapture(cfg.data_root)
        print(f"Total samples: {len(samples)}")
        
        rng = random.Random(cfg.seed)  
        indices = list(range(len(samples)))
        rng.shuffle(indices)
        
        split = int(len(samples) * cfg.train_ratio)
        train_indices = indices[:split]
        val_indices = indices[split:]
        
        print(f"Train samples: {len(train_indices)}, Val samples: {len(val_indices)}")
    else:
        samples = None
        train_indices = None
        val_indices = None
    
    # 广播所有必要数据
    samples = broadcast_object(samples, src=0)
    train_indices = broadcast_object(train_indices, src=0)
    val_indices = broadcast_object(val_indices, src=0)
    
    # 根据indices创建样本列表
    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]

    # Model & Loss (Wrapped in DDP)
    model = UnifiedAligner(cfg).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    criterion = DualAnchorContrastiveLoss().to(device)
    evaluator = DistributedMultiModalEvaluator(device=device, ks=[1, 10, 30])

    # Data
    dataset = XCaptureDataset(train_samples, '/data1/yizhuo/ULIA/image_descriptions.csv', cfg.target_sr)
    val_dataset = XCaptureDataset(val_samples, '/data1/yizhuo/ULIA/image_descriptions.csv', cfg.target_sr)

    clip_proc = CLIPProcessor.from_pretrained(cfg.clip_path)
    audio_proc = AutoProcessor.from_pretrained(cfg.qwen_processor_path).feature_extractor
    
    sampler = DistributedSampler(dataset, shuffle=True, seed=cfg.seed)
    val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=cfg.seed)

    loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, 
                        collate_fn=get_collate_fn(clip_proc, audio_proc), num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, sampler=val_sampler, 
                        collate_fn=get_collate_fn(clip_proc, audio_proc), num_workers=4, pin_memory=True)
    if rank == 0:
        print(f"train_loader {len(train_samples)} val_loader {len(val_samples)}")

    # Optimizer & Scaler & warm up
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr)

    total_steps = len(loader) * cfg.epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_fp16)

    if cfg.use_fp16 and device.type == "cuda":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.float32

    if rank == 0:
        wandb.init(project="MultiModal-Align", name="text-image-audio align")

    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
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
            scheduler.step()

            if rank == 0:
                wandb.log(loss_dict)
                pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")

        val_metrics, val_loss = evaluator.evaluate(model, val_loader)
            
        if rank == 0:
            print(f"Epoch {epoch} Metrics: {val_metrics}")
            wandb.log(val_metrics)
            wandb.log(val_loss)
        
        dist.barrier()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()