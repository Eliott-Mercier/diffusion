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

## Train

```bash
python train.py --config configs/baseline.yaml
python train.py --config configs/baseline.yaml --set name=trial lr=1e-3 image_size=28
```

Hyperparameters: `config.py`. Results: `runs/<name>_<date>/`.

## Evaluate and Analyze

```bash
python classifier.py --image_size 28            # judge, if classifier/mnist_cnn_28.pt is missing
python verifications.py                         # noising, schedule, overfitting, trajectory
python evaluate.py runs/<run>                   # confidence, diversity, % failed
python fid.py --references runs/<run>           # FID et IS
python compare.py runs/<a> runs/<b> --out <name> # table and comparison figures
python evolution.py runs/<run>                  # quality over epochs (checkpoint_every > 0)
python memorisation.py runs/<run>               # nearest neighbor in training set
python loss_par_t.py runs/<a> runs/<b> --out <name>
python visualiser.py runs/<run>                 # noising.png and trajectory.png
```

## Reproduce Experiments

```bash
bash experiences/serie1.sh   # ... serie5.sh, puis fid.sh, compare_serie2.sh, compare_phase2.sh
```

Final model: `runs/C2_C1+cosine_28_20261114-114722`.
