# DDPM sur MNIST

## Authors

- RABEMANANTSOA Clarck
- MERCIER DEL FORNO Eliott
- FAURE Tristan

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Entraîner

```bash
python train.py --config configs/baseline.yaml
python train.py --config configs/baseline.yaml --set name=essai lr=1e-3 image_size=28
```

Hyperparamètres : `config.py`. Résultats : `runs/<name>_<date>/`.

## Évaluer et analyser

```bash
python classifier.py --image_size 28            # juge, si classifier/mnist_cnn_28.pt manque
python verifications.py                         # bruitage, schedule, surapprentissage, trajectoire
python evaluate.py runs/<run>                   # confiance, diversité, % ratées
python fid.py --references runs/<run>           # FID et IS
python compare.py runs/<a> runs/<b> --out <nom> # tableau et figures de comparaison
python evolution.py runs/<run>                  # qualité au fil des époques (checkpoint_every > 0)
python memorisation.py runs/<run>               # plus proche voisin dans l'entraînement
python loss_par_t.py runs/<a> runs/<b> --out <nom>
python visualiser.py runs/<run>                 # bruitage.png et trajectoire.png
```

## Reproduire les essais

```bash
bash experiences/serie1.sh   # ... serie5.sh, puis fid.sh, compare_serie2.sh, compare_phase2.sh
```

Modèle final : `runs/C2_C1+cosine_28_20261114-114722`.
