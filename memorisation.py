import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from utils import generate, get_device, real_images, show


@torch.no_grad()
def nearest(x, train):
    x, train = x.flatten(1), train.flatten(1)
    best_d, best_i = [], []
    for b in x.split(250):
        d = torch.cdist(b, train)
        dmin, imin = d.min(1)
        best_d.append(dmin.cpu())
        best_i.append(imin.cpu())
    return torch.cat(best_d), torch.cat(best_i)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--weights", default="model.pt")
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()
    device = get_device()

    cfg, gen, _ = generate(args.run, args.n, device, weights=args.weights)
    train = real_images(cfg.image_size, True, 60000).to(device)
    test = real_images(cfg.image_size, False, args.n).to(device)

    d_gen, i_gen = nearest(gen, train)
    d_test, _ = nearest(test, train)
    print(f"distance médiane au plus proche voisin : générées {d_gen.median():.2f} | "
          f"vraies images de test {d_test.median():.2f}")

    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.4])
    ax = fig.add_subplot(gs[:, 0])
    ax.hist(d_test, bins=30, alpha=0.6, label=f"vraies images de test (méd. {d_test.median():.1f})")
    ax.hist(d_gen, bins=30, alpha=0.6, label=f"images générées (méd. {d_gen.median():.1f})")
    ax.set_xlabel("distance à l'image d'entraînement la plus proche"), ax.set_ylabel("nombre")
    ax.set_title("Proximité avec l'entraînement"), ax.legend(fontsize=8)

    order = d_gen.argsort()[:8]
    sub = gs[:, 1].subgridspec(2, 8)
    for j, k in enumerate(order):
        for row, (img, label) in enumerate([(gen[k], "générée"), (train[i_gen[k]], "plus proche\n(entraînement)")]):
            a = fig.add_subplot(sub[row, j])
            show(a, img[0])
            if row == 0:
                a.set_title(f"d={d_gen[k]:.1f}", fontsize=8)
            if j == 0:
                a.text(-0.2, 0.5, label, transform=a.transAxes, rotation=90, va="center", ha="right", fontsize=8)
    fig.suptitle(f"Mémorisation ? {Path(args.run).name} ({args.weights})")
    fig.tight_layout()
    out = Path("figures/memorisation")
    out.mkdir(parents=True, exist_ok=True)
    name = f"{Path(args.run).name}_{Path(args.weights).stem}.png"
    fig.savefig(out / name, dpi=120)
    print(f"Figure : {out / name}")


if __name__ == "__main__":
    main()
