import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from classifier import JUDGE_SIZE, load_classifier
from evaluate import judge
from utils import generate, get_device, read_losses, to_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()
    device = get_device()
    run = Path(args.run)
    out = Path("figures/evolution") / run.name
    out.mkdir(parents=True, exist_ok=True)

    loss = {int(r["epoch"]): float(r["loss"]) for r in read_losses(run)[1]}
    clf = load_classifier(JUDGE_SIZE, device)

    rows, grids = [], {}
    for ckpt in sorted((run / "checkpoints").glob("epoch_*.pt")):
        epoch = int(ckpt.stem.split("_")[1])
        _, imgs, _ = generate(run, args.n, device, weights=f"checkpoints/{ckpt.name}")
        m = judge(clf, imgs)[2]
        rows.append({"epoch": epoch, "loss": round(loss[epoch], 5), "confidence": m["mean_confidence"],
                 "pct_confident": m["pct_confident"], "pct_failed": m["pct_failed"],
                 "effective_digit_count": m["effective_digit_count"]})
        grids[epoch] = make_grid(to_img(imgs[:16].cpu()), nrow=8, pad_value=1)
        print(rows[-1])

    with open(out / "evolution.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    ep = [r["epoch"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(sorted(loss), [loss[e] for e in sorted(loss)], color="gray", label="loss (average over t)")
    ax1.set_xlabel("epoch"), ax1.set_ylabel("loss", color="gray"), ax1.set_yscale("log")
    ax2 = ax1.twinx()
    ax2.plot(ep, [r["confidence"] for r in rows], marker=".", color="tab:blue", label="judge confidence")
    ax2.plot(ep, [r["pct_confident"] / 100 for r in rows], marker=".", color="tab:orange", label="fraction > 0.9")
    ax2.set_ylabel("quality (judge)"), ax2.set_ylim(0, 1)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax1.grid(alpha=0.3)
    ax1.set_title(f"Loss and quality over epochs: {run.name}")
    fig.tight_layout()
    fig.savefig(out / "curve.png", dpi=120)
    plt.close(fig)

    show = [e for e in ep if e in (5, 10, 20, 50, 100, 150, 200)] or ep[:: max(1, len(ep) // 6)]
    fig, axes = plt.subplots(len(show), 1, figsize=(8, 2.2 * len(show)), squeeze=False)
    by_epoch = {r["epoch"]: r for r in rows}
    for ax, e in zip(axes[:, 0], show):
        r = by_epoch[e]
        ax.imshow(grids[e].permute(1, 2, 0))
        ax.set_title(f"epoch {e}: confidence {r['confidence']:.2f}, {r['pct_confident']}% > 0.9", fontsize=9)
        ax.axis("off")
    fig.suptitle("Same seeds, different epochs")
    fig.tight_layout()
    fig.savefig(out / "samples.png", dpi=120)
    plt.close(fig)
    print(f"Figures in {out}/")


if __name__ == "__main__":
    main()
