import os
import glob
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from peft import LoraConfig, get_peft_model
from PIL import Image

# =========================
# CUDA SETTINGS (important for Tesla P4 / sm_61)
# =========================

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# =========================
# EXR loader
# =========================

def load_exr(path):

    import imageio.v3 as iio

    try:
        # FORCE FreeImage backend (not OpenCV)
        d = iio.imread(path, plugin="freeimage")

    except Exception:
        # fallback to OpenEXR library directly
        import OpenEXR, Imath

        exr = OpenEXR.InputFile(path)

        dw = exr.header()['dataWindow']

        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1

        pt = Imath.PixelType(Imath.PixelType.FLOAT)

        raw = exr.channel("R", pt)

        d = np.frombuffer(raw, dtype=np.float32).reshape(h, w)

    if d.ndim == 3:
        d = d[...,0]

    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    return d.astype(np.float32)

# =========================
# DATASET
# =========================

class DepthDataset(Dataset):

    def __init__(self, img_dir, depth_dir, processor):

        self.processor = processor

        self.images = sorted(glob.glob(os.path.join(img_dir,"*.png")))

        self.depths = [
            os.path.join(depth_dir, os.path.basename(p).replace(".png",".exr"))
            for p in self.images
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):

        img = Image.open(self.images[i]).convert("RGB")

        depth = load_exr(self.depths[i])

        depth = np.nan_to_num(depth)

        mask = depth > 0

        inputs = self.processor(images=img, return_tensors="pt")

        return {
            "pixel_values": inputs.pixel_values[0],
            "depth": torch.from_numpy(depth).float(),
            "mask": torch.from_numpy(mask.astype(np.float32))
        }

# =========================
# COLLATE
# =========================

def collate(batch):

    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "depth": [b["depth"] for b in batch],
        "mask": [b["mask"] for b in batch]
    }

# =========================
# LOSS
# =========================

def ssi_loss(pred, gt, mask):

    pred = pred.squeeze(1)

    losses = []

    for i in range(pred.shape[0]):

        valid = mask[i] > 0.5

        if valid.sum() < 10:
            continue

        p = pred[i][valid]
        g = gt[i][valid]

        A00 = (p*p).sum()
        A01 = p.sum()
        A11 = valid.sum()

        B0 = (p*g).sum()
        B1 = g.sum()

        det = A00*A11 - A01*A01

        if abs(det) < 1e-6:

            s = torch.tensor(1.0, device=p.device)
            t = torch.tensor(0.0, device=p.device)

        else:

            s = (A11*B0 - A01*B1)/det
            t = (-A01*B0 + A00*B1)/det

        losses.append(torch.mean(torch.abs(s*p + t - g)))

    # CRITICAL FIX: keep gradient graph valid
    if len(losses) == 0:

        return pred.mean() * 0.0

    return torch.mean(torch.stack(losses))

# =========================
# TRAIN
# =========================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./train_output")

    parser.add_argument("--out", default="./depth_anything_lora")

    parser.add_argument("--epochs", type=int, default=2)

    parser.add_argument("--batch", type=int, default=4)

    args = parser.parse_args()

    device = "cuda"

    print("Loading model...")

    processor = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Base-hf"
    )

    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Base-hf"
    )

    # =========================
    # LoRA config
    # =========================

    config = LoraConfig(

        r=16,
        lora_alpha=32,
        target_modules=["query","value"],
        lora_dropout=0.05,
        bias="none"
    )

    model = get_peft_model(model, config)

    model = model.to(device)

    print("Trainable params:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    dataset = DepthDataset(

        os.path.join(args.data_dir,"images"),
        os.path.join(args.data_dir,"depth"),
        processor
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        collate_fn=collate,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4
    )

    scaler = torch.cuda.amp.GradScaler()

    model.train()

    for epoch in range(args.epochs):

        print("Epoch",epoch+1)

        for i,batch in enumerate(loader):

            pixel_values = batch["pixel_values"].to(device)

            with torch.cuda.amp.autocast(dtype=torch.float16):

                out = model(pixel_values=pixel_values)

                pred = out.predicted_depth

                gt = torch.stack([
                    F.interpolate(
                        d.unsqueeze(0).unsqueeze(0),
                        pred.shape[-2:],
                        mode="nearest"
                    )[0,0]
                    for d in batch["depth"]
                ]).to(device)

                mask = torch.stack([
                    F.interpolate(
                        m.unsqueeze(0).unsqueeze(0),
                        pred.shape[-2:],
                        mode="nearest"
                    )[0,0]
                    for m in batch["mask"]
                ]).to(device)

                loss = ssi_loss(pred, gt, mask)

            optimizer.zero_grad()

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            if i%50==0:
                print("step",i,"loss",loss.item())

    print("Saving model...")

    model.save_pretrained(args.out)

    processor.save_pretrained(args.out)

    print("DONE")

# =========================

if __name__=="__main__":
    main()
