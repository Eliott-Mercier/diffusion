import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from config import load_config
from diffusion import Diffusion
from model import build_model
from utils import get_dataset, get_device, power_w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)

    torch.manual_seed(cfg.seed)
    device = get_device()
    run_dir = Path("runs") / f"{cfg.name}_{time.strftime('%Y%m%d-%H%M%S')}"
    (run_dir / "samples").mkdir(parents=True)
    cfg.save(run_dir / "config.yaml")
    watts = power_w(cfg.power_w, device)
    print(f"Run : {run_dir} | device : {device} | puissance supposée {watts:.0f} W")

    dataset = get_dataset(cfg.image_size)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    diffusion = Diffusion(cfg.T, cfg.schedule, cfg.beta_start, cfg.beta_end, device)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    print(f"alpha_bar final = {diffusion.alpha_bar[-1]:.5f}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params:,} paramètres")

    losses = []
    start = time.time()
    train_s = 0.0  # sans le temps de génération des images de suivi
    with open(run_dir / "losses.csv", "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "time_s", "train_s", "energy_wh"])

    for epoch in range(1, cfg.epochs + 1):
        step_losses = []
        epoch_start = time.time()
        for x0, _ in loader:
            x0 = x0.to(device)
            loss = diffusion.loss(model, x0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_losses.append(loss.detach())

        losses.append(sum(torch.stack(step_losses).tolist()) / len(loader))
        train_s += time.time() - epoch_start
        elapsed = time.time() - start
        energy_wh = elapsed * watts / 3600
        print(f"Époque {epoch}/{cfg.epochs}  loss {losses[-1]:.4f}  ({elapsed / 60:.1f} min, {energy_wh:.2f} Wh)")
        with open(run_dir / "losses.csv", "a", newline="") as f:
            csv.writer(f).writerow([epoch, losses[-1], round(elapsed, 1), round(train_s, 1), round(energy_wh, 3)])

        if epoch % cfg.sample_every == 0 or epoch == cfg.epochs:
            imgs = diffusion.sample(model, cfg.n_samples, cfg.image_size, device)
            save_image((imgs + 1) / 2, run_dir / "samples" / f"epoch_{epoch:03d}.png", nrow=8)
            torch.save(model.state_dict(), run_dir / "model.pt")
        if cfg.checkpoint_every and epoch % cfg.checkpoint_every == 0:
            (run_dir / "checkpoints").mkdir(exist_ok=True)
            torch.save(model.state_dict(), run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")

    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("époque"), plt.ylabel("loss (MSE sur le bruit)"), plt.title(cfg.name)
    plt.savefig(run_dir / "loss.png", dpi=120)

    total_s = time.time() - start
    summary = {
        "device": device,
        "power_w": watts,
        "n_params": n_params,
        "alpha_bar_final": round(diffusion.alpha_bar[-1].item(), 6),
        "final_loss": round(losses[-1], 5),
        "train_time_s": round(train_s, 1),
        "total_time_s": round(total_s, 1),
        "energy_wh": round(total_s * watts / 3600, 3),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Terminé — résultats dans {run_dir} | {total_s / 60:.1f} min, ~{summary['energy_wh']:.2f} Wh")


if __name__ == "__main__":
    main()
