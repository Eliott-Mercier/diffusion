import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy import linalg

from classifier import JUDGE_SIZE, load_classifier, to_judge
from evaluate import ratees
from utils import generate, get_device, read_json, real_images

N_REF = 10000
OUT = Path("figures/fid")


@torch.no_grad()
def feats_and_probs(clf, imgs):
    feats, probs = [], []
    for b in to_judge(imgs).split(500):
        f = clf.features(b)
        feats.append(f.cpu().double())
        probs.append(clf.fc2(f).softmax(1).cpu().double())
    return torch.cat(feats).numpy(), torch.cat(probs).numpy()


def gaussian(feats):
    return feats.mean(0), np.cov(feats, rowvar=False)


def fid(stats_a, stats_b):
    (mu1, s1), (mu2, s2) = stats_a, stats_b
    covmean = linalg.sqrtm(s1 @ s2).real  # partie imaginaire = erreurs d'arrondi
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def inception_score(probs):
    py = probs.mean(0, keepdims=True)
    kl = (probs * (np.log(probs + 1e-12) - np.log(py + 1e-12))).sum(1)
    return float(np.exp(kl.mean()))


def reference_stats(clf, device):
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "ref_stats.npz"
    if cache.exists():
        z = np.load(cache)
        return z["mu"], z["sigma"]
    f, _ = feats_and_probs(clf, real_images(JUDGE_SIZE, True, N_REF).to(device))
    mu, sigma = gaussian(f)
    np.savez(cache, mu=mu, sigma=sigma)
    return mu, sigma


def references(clf, device, ref):
    sources = [("vraies images 28x28 (test)", real_images(28, False, N_REF)),
               ("vraies images 16x16 agrandies (test)", real_images(16, False, N_REF))]
    torch.manual_seed(0)
    sources.append(("bruit pur N(0, I)", torch.randn(N_REF, 1, 28, 28, device=device)))
    out = {}
    for name, imgs in sources:
        f, p = feats_and_probs(clf, imgs.to(device))
        out[name] = {"fid": round(fid(ref, gaussian(f)), 2), "is": round(inception_score(p), 2)}
        print(f"{name:40s} FID {out[name]['fid']:8.2f} | IS {out[name]['is']:.2f}")
    (OUT / "references.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def evaluate_fid(run_dir, clf, device, n, ref):
    _, imgs, gen_time = generate(run_dir, n, device)
    f, p = feats_and_probs(clf, imgs)
    res = {
        "fid": round(fid(ref, gaussian(f)), 2),
        "is": round(inception_score(p), 2),
        "pct_ratees": round(ratees(imgs).float().mean().item() * 100, 1),
        "n": n,
        "temps_generation_s": round(gen_time, 1),
        "protocole": f"classifieur MNIST {JUDGE_SIZE}x{JUDGE_SIZE} (128 caractéristiques), réf. = {N_REF} images d'entraînement",
    }
    (Path(run_dir) / "fid.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"{Path(run_dir).name:45s} FID {res['fid']:8.2f} | IS {res['is']:.2f} | "
          f"{res['pct_ratees']}% ratées | {gen_time:.0f}s")
    return res


def table():
    rows = []
    refs = OUT / "references.json"
    if refs.exists():
        rows += [[name, v["fid"], v["is"], "", "", ""] for name, v in json.loads(refs.read_text()).items()]
    runs = []
    for fj in sorted(Path("runs").glob("*/fid.json")):
        r = json.loads(fj.read_text())
        ev = read_json(fj.parent / "eval.json")
        runs.append([fj.parent.name, r["fid"], r["is"], ev.get("confiance_moyenne", ""),
                     ev.get("nb_chiffres_effectif", ""), r["pct_ratees"]])
    rows += sorted(runs, key=lambda r: r[1])
    with open(OUT / "tableau.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "fid", "is", "confiance", "nb_chiffres_effectif", "pct_ratees"])
        w.writerows(rows)
    for r in rows:
        print(" | ".join(str(x) for x in r))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--references", action="store_true")
    args = parser.parse_args()
    device = get_device()
    clf = load_classifier(JUDGE_SIZE, device)
    ref = reference_stats(clf, device)
    if args.references:
        references(clf, device, ref)
    for run_dir in args.runs:
        evaluate_fid(run_dir, clf, device, args.n, ref)
    table()


if __name__ == "__main__":
    main()
