import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from utils import load_run, real_images, show

FRACTIONS = (0, 0.05, 0.1, 0.25, 0.5, 0.75, 1)


def steps(T):
    return sorted({min(int(f * T), T - 1) for f in FRACTIONS})


def bruitage(run_dir, cfg, d):
    x0 = real_images(cfg.image_size, True, 4)
    ts = steps(cfg.T)
    torch.manual_seed(0)
    fig, axes = plt.subplots(4, len(ts), figsize=(1.4 * len(ts), 6.4))
    for j, t in enumerate(ts):
        xt, _ = d.add_noise(x0, torch.full((4,), t))
        for i in range(4):
            show(axes[i, j], xt[i, 0])
        axes[0, j].set_title(f"t={t}\n" + r"$\bar\alpha$" + f"={d.alpha_bar[t]:.3f}", fontsize=8)
    fig.suptitle(f"Bruitage ({cfg.schedule}, T={cfg.T}) : {run_dir.name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(run_dir / "bruitage.png", dpi=110)
    plt.close(fig)


@torch.no_grad()
def trajectoire(run_dir, cfg, model, d, n=6):
    ts = set(steps(cfg.T)) | {int(cfg.T * 0.2), int(cfg.T * 0.3)}
    torch.manual_seed(0)
    x = torch.randn(n, 1, cfg.image_size, cfg.image_size)
    snaps = {}
    for t in reversed(range(cfg.T)):
        x = d.denoise_step(model, x, t)
        if t in ts:
            snaps[t] = x.clamp(-1, 1)
    keep = sorted(snaps, reverse=True)
    fig, axes = plt.subplots(n, len(keep), figsize=(1.3 * len(keep), 1.4 * n))
    for j, t in enumerate(keep):
        for i in range(n):
            show(axes[i, j], snaps[t][i, 0])
        axes[0, j].set_title(f"t={t}", fontsize=8)
    fig.suptitle(f"Génération pas à pas : {run_dir.name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(run_dir / "trajectoire.png", dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    runs = [Path(r) for r in args.runs]
    if args.all:
        runs = [p.parent for p in sorted(Path("runs").glob("*/model.pt"))]
    for run_dir in runs:
        cfg, model, d = load_run(run_dir, args.device)
        bruitage(run_dir, cfg, d)
        trajectoire(run_dir, cfg, model, d)
        print(f"{run_dir.name} : bruitage.png, trajectoire.png")


if __name__ == "__main__":
    main()
