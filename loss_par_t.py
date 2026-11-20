import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from utils import get_device, load_run, real_images


@torch.no_grad()
def loss_by_t(run_dir, device, n_bins=25, n_images=512):
    cfg, model, d = load_run(run_dir, device)
    x0 = real_images(cfg.image_size, False, n_images).to(device)
    torch.manual_seed(0)
    ts = torch.linspace(0, cfg.T - 1, n_bins).long()
    losses = []
    for t in ts:
        tt = torch.full((n_images,), int(t), device=device)
        xt, eps = d.add_noise(x0, tt)
        losses.append(F.mse_loss(model(xt, tt), eps).item())
    return (ts.float() / cfg.T).numpy(), losses, d.alpha_bar[ts.to(device)].cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--labels", nargs="*", default=None)
    args = parser.parse_args()
    device = get_device()
    labels = args.labels or [Path(r).name for r in args.runs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for run_dir, label in zip(args.runs, labels):
        x, losses, ab = loss_by_t(run_dir, device)
        ax1.plot(x, losses, marker=".", label=f"{label} (moyenne {sum(losses) / len(losses):.3f})")
        ax2.plot(ab, losses, marker=".", label=label)
    ax1.set_xlabel("t / T"), ax1.set_ylabel("MSE sur le bruit"), ax1.set_yscale("log")
    ax1.set_title("Loss selon l'étape t"), ax1.grid(alpha=0.3), ax1.legend(fontsize=8)
    ax2.set_xlabel(r"$\bar\alpha_t$ (part de l'image restante)"), ax2.set_yscale("log")
    ax2.set_title(r"Loss selon le niveau de bruit $\bar\alpha_t$"), ax2.grid(alpha=0.3), ax2.legend(fontsize=8)
    fig.tight_layout()
    out = Path("figures/loss_par_t")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{args.out}.png", dpi=120)
    print(f"Figure : {out / args.out}.png")


if __name__ == "__main__":
    main()
