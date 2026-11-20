import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from config import load_config
from diffusion import Diffusion
from model import build_model
from utils import get_device, load_run, real_images, show


def show_row(axes, imgs, titles):
    for ax, img, title in zip(axes, imgs, titles):
        show(ax, img[0, 0])
        ax.set_title(title, fontsize=8)


def check_schedule(cfg, out):
    cases = [
        ("linear T=1000 (baseline)", dict(T=1000, schedule="linear", beta_end=0.02)),
        ("cosine T=1000", dict(T=1000, schedule="cosine", beta_end=0.02)),
        ("linear T=200, beta_end=0.05 (lab)", dict(T=200, schedule="linear", beta_end=0.05)),
        ("linear T=100, beta_end=0.02 (bad)", dict(T=100, schedule="linear", beta_end=0.02)),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    print("final alpha_bar:")
    for label, p in cases:
        d = Diffusion(p["T"], p["schedule"], cfg.beta_start, p["beta_end"])
        ax.plot(torch.arange(p["T"]) / p["T"], d.alpha_bar, label=label)
        print(f"  {label:40s} {d.alpha_bar[-1]:.5f}")
    ax.set_xlabel("t / T"), ax.set_ylabel(r"$\bar\alpha_t$ (image fraction in $x_t$)")
    ax.set_title(r"Schedules: $\bar\alpha_t$ should end close to 0")
    ax.grid(alpha=0.3), ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "1_schedule.png", dpi=120)
    plt.close(fig)


def check_noising(cfg, out):
    x0 = real_images(cfg.image_size, True, 4)
    steps = [0, 50, 100, 250, 500, 750, cfg.T - 1]
    for schedule in ("linear", "cosine"):
        d = Diffusion(cfg.T, schedule, cfg.beta_start, cfg.beta_end)
        fig, axes = plt.subplots(len(x0), len(steps), figsize=(1.4 * len(steps), 1.6 * len(x0)))
        torch.manual_seed(cfg.seed)
        for j, t in enumerate(steps):
            xt, _ = d.add_noise(x0, torch.full((len(x0),), t))
            for i in range(len(x0)):
                show(axes[i, j], xt[i, 0])
            axes[0, j].set_title(f"t={t}\n" + r"$\bar\alpha$" + f"={d.alpha_bar[t]:.3f}", fontsize=8)
        fig.suptitle(f"Forward process (noising), schedule {schedule}")
        fig.tight_layout()
        fig.savefig(out / f"2_noising_{schedule}.png", dpi=120)
        plt.close(fig)


def check_overfitting(cfg, device, out, steps=1500):
    x0 = real_images(cfg.image_size, True, cfg.batch_size).to(device)
    d = Diffusion(cfg.T, cfg.schedule, cfg.beta_start, cfg.beta_end, device)

    histories = {}
    for mode in ("A: fixed x0, t and noise", "B: fixed x0, random t and noise"):
        torch.manual_seed(cfg.seed)
        model = build_model(cfg).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        t_fix = torch.randint(0, cfg.T, (len(x0),), device=device)
        xt_fix, eps_fix = d.add_noise(x0, t_fix)
        losses = []
        for _ in range(steps):
            if mode.startswith("A"):
                loss = F.mse_loss(model(xt_fix, t_fix), eps_fix)
            else:
                loss = d.loss(model, x0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        histories[mode] = losses
        print(f"  {mode:40s} final loss {sum(losses[-50:]) / 50:.5f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    for mode, losses in histories.items():
        ax.plot(losses, label=mode, alpha=0.8)
    ax.axhline(0.041, color="gray", ls="--", label="baseline (60,000 images, 20 epochs)")
    ax.set_yscale("log"), ax.set_xlabel("optimization steps"), ax.set_ylabel("loss (MSE, log)")
    ax.set_title(f"Overfitting on a single mini-batch ({len(x0)} images)")
    ax.grid(alpha=0.3), ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "3_surapprentissage.png", dpi=120)
    plt.close(fig)


def check_trajectory(run_dir, device, out, n=6):
    cfg, model, d = load_run(run_dir, device)
    keep = sorted({cfg.T - 1, *[int(cfg.T * f) for f in (0.9, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05)], 0}, reverse=True)
    torch.manual_seed(cfg.seed)
    x = torch.randn(n, 1, cfg.image_size, cfg.image_size, device=device)
    snaps = {}
    for t in reversed(range(cfg.T)):
        x = d.denoise_step(model, x, t)
        if t in keep:
            snaps[t] = x.clamp(-1, 1).cpu()
    fig, axes = plt.subplots(n, len(keep), figsize=(1.3 * len(keep), 1.4 * n))
    for j, t in enumerate(keep):
        for i in range(n):
            show(axes[i, j], snaps[t][i, 0])
        axes[0, j].set_title(f"t={t}", fontsize=8)
    fig.suptitle(f"Reverse process (generation), {Path(run_dir).name}")
    fig.tight_layout()
    fig.savefig(out / "4_trajectory.png", dpi=120)
    plt.close(fig)


def check_prediction_x0(run_dir, device, out):
    cfg, model, d = load_run(run_dir, device)
    x0 = real_images(cfg.image_size, False, 1).to(device)
    ts = [int(cfg.T * f) for f in (0.01, 0.1, 0.25, 0.5, 0.75)] + [cfg.T - 1]
    torch.manual_seed(cfg.seed)
    fig, axes = plt.subplots(2, len(ts) + 1, figsize=(1.4 * (len(ts) + 1), 3.7))
    show_row(axes[:, 0], [x0, x0], ["original", ""])
    for j, t in enumerate(ts, start=1):
        tt = torch.tensor([t], device=device)
        xt, _ = d.add_noise(x0, tt)
        show_row(axes[:, j], [xt, d.predict_x0(model, xt, tt)], [f"$x_t$, t={t}", r"predicted $\hat{x}_0$"])
    fig.suptitle("One-step estimation of $x_0$ from $x_t$")
    fig.tight_layout()
    fig.savefig(out / "5_prediction_x0.png", dpi=120)
    plt.close(fig)


CHECKS = ["schedule", "noising", "overfitting", "trajectory", "prediction"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--run", default="runs/baseline_20261111-151624")
    parser.add_argument("--only", nargs="*", default=CHECKS, choices=CHECKS)
    parser.add_argument("--out", default="figures/verifications")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device: {device} | figures in {out}/")

    for name in args.only:
        print(f"[{name}]")
        if name == "schedule":
            check_schedule(cfg, out)
        elif name == "noising":
            check_noising(cfg, out)
        elif name == "overfitting":
            check_overfitting(cfg, device, out)
        elif name == "trajectory":
            check_trajectory(args.run, device, out)
        elif name == "prediction":
            check_prediction_x0(args.run, device, out)


if __name__ == "__main__":
    main()
