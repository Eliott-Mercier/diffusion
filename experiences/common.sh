set -e
B=runs/baseline_20261111-172332
C=(image_size=28 epochs=50 checkpoint_every=10 sample_every=10 model=resunet lr=1e-3)
run() { python train.py --config configs/baseline.yaml --set "$@"; }
last() { ls -d runs/$1_2* | tail -1; }
