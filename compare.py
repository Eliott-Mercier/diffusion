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

# sans effet sur le modèle
IGNORE = {"name", "sample_every", "n_samples", "power_w"}


def changes(cfg):
    default, values = asdict(Config()), asdict(cfg)
    diff = [f"{k}={v}" for k, v in values.items() if k not in IGNORE and v != default[k]]
    return ", ".join(diff) or "baseline"


def samples_grid(run_dir, device):
    imgs = generate(run_dir, 16, device)[1]  # même graine pour tous les runs
    return make_grid(to_img(imgs.cpu()), nrow=8, pad_value=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", required=True, help="nom de la comparaison (dossier de sortie)")
    parser.add_argument("--labels", nargs="*", default=None, help="un label par run (défaut : ce qui change)")
    args = parser.parse_args()

    device = get_device()
    out = Path("figures/comparaisons") / args.out
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
            "loss finale": round(losses[-1], 4),
            "époques": len(losses),
            "temps entr. (min)": round(float(rows[-1]["time_s"]) / 60, 1),
            "device": summary.get("device", "?"),
            "énergie (Wh)": summary.get("energy_wh", "?"),
            "génération 1000 img (s)": ev["temps_generation_s"],
            "confiance": ev["confiance_moyenne"],
            "% > 0.9": ev["pct_confiantes"],
            "% ratées": ev.get("pct_ratees", "?"),
            "nb chiffres eff.": ev.get("nb_chiffres_effectif", "?"),
            "FID": fd.get("fid", "-"),
            "IS": fd.get("is", "-"),
            "chiffre min/max %": f"{ev['chiffre_min_pct']} / {ev['chiffre_max_pct']}",
            "grid": samples_grid(run_dir, device),
        })

    fig, ax = plt.subplots(figsize=(7, 4))
    for r in runs:
        ax.plot(range(1, len(r["losses"]) + 1), r["losses"], marker=".", label=r["label"])
    ax.set_yscale("log"), ax.set_xlabel("époque"), ax.set_ylabel("loss (MSE sur le bruit, log)")
    ax.set_title(f"Comparaison : {args.out}"), ax.grid(alpha=0.3), ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "loss.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(len(runs), 1, figsize=(8, 2.3 * len(runs)), squeeze=False)
    for ax, r in zip(axes[:, 0], runs):
        ax.imshow(r["grid"].permute(1, 2, 0))
        ax.set_title(f"{r['label']}   (confiance {r['confiance']:.2f}, {r['% > 0.9']}% > 0.9)", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Échantillons, même graine : {args.out}")
    fig.tight_layout()
    fig.savefig(out / "samples.png", dpi=120)
    plt.close(fig)

    cols = ["label", "loss finale", "époques", "temps entr. (min)", "device", "énergie (Wh)",
            "génération 1000 img (s)", "confiance", "% > 0.9", "% ratées", "nb chiffres eff.", "FID", "IS", "chiffre min/max %", "run"]
    with open(out / "tableau.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows([[r[c] for c in cols] for r in runs])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in runs]
    print("\n".join(lines))
    print(f"\nFigures dans {out}/")


if __name__ == "__main__":
    main()
