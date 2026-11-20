import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from classifier import JUDGE_SIZE, load_classifier, to_judge
from utils import generate, get_device, real_images, show

SEUIL = 0.9
SEUIL_FOND = 0.5  # part minimale de pixels noirs (fond)


@torch.no_grad()
def scores(clf, imgs):
    probs = torch.cat([clf(b).softmax(1) for b in to_judge(imgs).split(500)])
    conf, pred = probs.max(1)
    return pred.cpu(), conf.cpu()


def ratees(imgs):
    return ((imgs < -0.5).float().mean((1, 2, 3)) < SEUIL_FOND).cpu()


def metrics(pred, conf, failed):
    counts = torch.bincount(pred[~failed], minlength=10)
    n = max(counts.sum().item(), 1)
    p = counts[counts > 0] / n
    return {
        "pct_ratees": round(failed.float().mean().item() * 100, 1),
        "confiance_moyenne": round(conf.mean().item(), 4),
        "pct_confiantes": round((conf > SEUIL).float().mean().item() * 100, 1),
        "repartition": counts.tolist(),
        "nb_chiffres_effectif": round(torch.exp(-(p * p.log()).sum()).item(), 2),
        "chiffre_min_pct": round(counts.min().item() / n * 100, 1),
        "chiffre_max_pct": round(counts.max().item() / n * 100, 1),
    }


def judge(clf, imgs):
    pred, conf = scores(clf, imgs)
    failed = ratees(imgs)
    conf = torch.where(failed, torch.zeros_like(conf), conf)
    return pred, conf, metrics(pred, conf, failed)


def figure(run_dir, imgs, pred, conf, m, ref):
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    x = torch.arange(10)
    ax.bar(x - 0.2, ref["repartition"], 0.4, label="vraies images", color="lightgray")
    ax.bar(x + 0.2, m["repartition"], 0.4, label="générées")
    ax.set_xticks(x), ax.set_xlabel("chiffre prédit"), ax.set_ylabel("nombre")
    ax.set_title("Répartition des chiffres"), ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(conf, bins=20, range=(0, 1))
    ax.axvline(SEUIL, color="red", ls="--", label=f"seuil {SEUIL}")
    ax.set_xlabel("confiance du classifieur"), ax.set_ylabel("nombre d'images")
    ax.set_title(f"Confiance (moyenne {m['confiance_moyenne']:.3f})"), ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    lignes = [f"{'':18s}{'générées':>10s}{'réf.':>8s}"]
    for label, key, fmt in [("confiance moy.", "confiance_moyenne", ".3f"), ("% conf. > 0.9", "pct_confiantes", ".1f"),
                            ("chiffre min %", "chiffre_min_pct", ".1f"), ("chiffre max %", "chiffre_max_pct", ".1f"),
                            ("nb chiffres eff.", "nb_chiffres_effectif", ".2f"), ("% ratées", "pct_ratees", ".1f")]:
        lignes.append(f"{label:18s}{m[key]:>10{fmt}}{ref[key]:>8{fmt}}")
    ax.text(0, 0.9, "\n".join(lignes), va="top", family="monospace", fontsize=9)

    imgs = imgs.cpu()
    order = conf.argsort()
    sub = gs[1, :].subgridspec(2, 8)
    for row, (title, idx) in enumerate([("les plus reconnues", order[-8:].flip(0)),
                                        ("les moins reconnues", order[:8])]):
        for j, i in enumerate(idx):
            a = fig.add_subplot(sub[row, j])
            show(a, imgs[i, 0])
            a.set_title(f"{pred[i].item()} ({conf[i]:.2f})", fontsize=8)
            if j == 0:
                a.text(-0.3, 0.5, title, transform=a.transAxes, rotation=90, va="center", ha="right", fontsize=8)
    fig.suptitle(f"Évaluation par classifieur : {Path(run_dir).name}  (n={len(imgs)})")
    fig.tight_layout()
    fig.savefig(Path(run_dir) / "eval.png", dpi=110)
    plt.close(fig)


def evaluate_run(run_dir, n=1000, device=None, clf=None):
    device = device or get_device()
    clf = clf or load_classifier(JUDGE_SIZE, device)
    cfg, imgs, gen_time = generate(run_dir, n, device)

    ref = judge(clf, real_images(cfg.image_size, False, n).to(device))[2]
    ref28 = ref if cfg.image_size == JUDGE_SIZE else judge(clf, real_images(JUDGE_SIZE, False, n).to(device))[2]
    pred, conf, m = judge(clf, imgs)
    m.update({"temps_generation_s": round(gen_time, 1), "n": n, "reference_vraies_images": ref,
              "reference_28_natif": ref28, "juge": f"classifieur {JUDGE_SIZE}x{JUDGE_SIZE}"})

    (Path(run_dir) / "eval.json").write_text(json.dumps(m, indent=2))
    figure(run_dir, imgs, pred, conf, m, ref)
    print(f"{Path(run_dir).name}: confiance {m['confiance_moyenne']:.3f} (réf. {ref['confiance_moyenne']:.3f}) | "
          f"{m['pct_confiantes']}% > {SEUIL} (réf. {ref['pct_confiantes']}%) | {m['pct_ratees']}% ratées | "
          f"{m['nb_chiffres_effectif']} chiffres eff. | répartition {m['repartition']} | génération {gen_time:.0f}s")
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()
    device = get_device()
    clf = load_classifier(JUDGE_SIZE, device)
    for run_dir in args.runs:
        evaluate_run(run_dir, args.n, device, clf)


if __name__ == "__main__":
    main()
