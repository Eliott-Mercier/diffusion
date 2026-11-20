import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from config import Config, load_config
from evaluate import evaluate_run
from utils import generate, get_device, read_json, read_losses, to_img

# No effect on model behavior.
IGNORE = {"name", "sample_every", "n_samples", "power_w"}


def changes(cfg):
    default, values = asdict(Config()), asdict(cfg)
    diff = [f"{k}={v}" for k, v in values.items() if k not in IGNORE and v != default[k]]
    return ", ".join(diff) or "baseline"


def samples_grid(run_dir, device):
    imgs = generate(run_dir, 16, device)[1]  # Same seed for all runs.
    return make_grid(to_img(imgs.cpu()), nrow=8, pad_value=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", required=True, help="comparison name (output directory)")
    parser.add_argument("--labels", nargs="*", default=None, help="one label per run (default: changed parameters)")
    args = parser.parse_args()

    device = get_device()
    out = Path("figures/comparisons") / args.out
    out.mkdir(parents=True, exist_ok=True)

    runs = []
    for i, run_dir in enumerate(args.runs):
        run_dir = Path(run_dir)
        cfg = load_config(run_dir / "config.yaml")
        losses, rows = read_losses(run_dir)
        ev = read_json(run_dir / "eval.json") or evaluate_run(run_dir, device=device)
        summary = read_json(run_dir / "summary.json")
        fd = read_json(run_dir / "fid.json")
        runs.append({
            "label": args.labels[i] if args.labels else changes(cfg),
            "run": run_dir.name,
            "losses": losses,
            "final loss": round(losses[-1], 4),
            "epochs": len(losses),
            "train time (min)": round(float(rows[-1]["time_s"]) / 60, 1),
            "device": summary.get("device", "?"),
            "energy (Wh)": summary.get("energy_wh", "?"),
            "generation 1000 img (s)": ev["generation_time_s"],
            "confidence": ev["mean_confidence"],
            "% > 0.9": ev["pct_confident"],
            "% failed": ev.get("pct_failed", "?"),
            "effective digits": ev.get("effective_digit_count", "?"),
            "FID": fd.get("fid", "-"),
            "IS": fd.get("is", "-"),
            "digit min/max %": f"{ev['digit_min_pct']} / {ev['digit_max_pct']}",
            "grid": samples_grid(run_dir, device),
        })

    fig, ax = plt.subplots(figsize=(7, 4))
    for r in runs:
        ax.plot(range(1, len(r["losses"]) + 1), r["losses"], marker=".", label=r["label"])
    ax.set_yscale("log"), ax.set_xlabel("epoch"), ax.set_ylabel("loss (noise MSE, log)")
    ax.set_title(f"Comparison: {args.out}"), ax.grid(alpha=0.3), ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "loss.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(len(runs), 1, figsize=(8, 2.3 * len(runs)), squeeze=False)
    for ax, r in zip(axes[:, 0], runs):
        ax.imshow(r["grid"].permute(1, 2, 0))
        ax.set_title(f"{r['label']}   (confidence {r['confidence']:.2f}, {r['% > 0.9']}% > 0.9)", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Samples, same seed: {args.out}")
    fig.tight_layout()
    fig.savefig(out / "samples.png", dpi=120)
    plt.close(fig)

    cols = ["label", "final loss", "epochs", "train time (min)", "device", "energy (Wh)",
            "generation 1000 img (s)", "confidence", "% > 0.9", "% failed", "effective digits", "FID", "IS", "digit min/max %", "run"]
    with open(out / "table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows([[r[c] for c in cols] for r in runs])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in runs]
    print("\n".join(lines))
    print(f"\nFigures in {out}/")


if __name__ == "__main__":
    main()
