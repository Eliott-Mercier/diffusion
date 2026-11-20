import csv
import json
import time
from pathlib import Path

import torch
from torchvision import datasets, transforms

from config import load_config
from diffusion import Diffusion
from model import build_model

# puissance supposée (W), pas mesurée : energie = duree x puissance
DEFAULT_POWER_W = {"mps": 40.0, "cpu": 45.0, "cuda": 70.0}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def power_w(cfg_power, device):
    return cfg_power if cfg_power > 0 else DEFAULT_POWER_W[device]


def get_dataset(image_size, train=True):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # [-1, 1], comme le bruit
    ])
    return datasets.MNIST("data", train=train, download=True, transform=transform)


def real_images(size, train, n):
    ds = get_dataset(size, train=train)
    if size == 28:
        return (ds.data[:n].unsqueeze(1).float().div(255) - 0.5) / 0.5
    return torch.stack([ds[i][0] for i in range(n)])


def to_img(x):
    return (x.clamp(-1, 1) + 1) / 2


def show(ax, img):
    ax.imshow(to_img(img).cpu(), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")


def read_json(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def read_losses(run_dir):
    with open(Path(run_dir) / "losses.csv") as f:
        rows = list(csv.DictReader(f))
    return [float(r["loss"]) for r in rows], rows


def load_run(run_dir, device, weights="model.pt"):
    cfg = load_config(Path(run_dir) / "config.yaml")
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(Path(run_dir) / weights, map_location=device))
    model.eval()
    return cfg, model, Diffusion(cfg.T, cfg.schedule, cfg.beta_start, cfg.beta_end, device)


def generate(run_dir, n, device, seed=0, weights="model.pt"):
    cfg, model, d = load_run(run_dir, device, weights)
    torch.manual_seed(seed)
    start = time.time()
    imgs = torch.cat([d.sample(model, b.numel(), cfg.image_size, device)
                      for b in torch.arange(n).split(500)])
    if device == "mps":
        torch.mps.synchronize()
    return cfg, imgs, time.time() - start
