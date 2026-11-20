#!/bin/bash
source "$(dirname "$0")/common.sh"
python compare.py $B $(last seed1) $(last seed2) --out variabilite --labels "seed 0" "seed 1" "seed 2"
python compare.py $B $(last cosine) --out schedule --labels "linear (baseline)" "cosine"
python compare.py $B $(last T100_beta0.02) --out T100_echec --labels "T=1000" "T=100, beta_end=0.02 (non adapté)"
python compare.py $(last epochs5) $B $(last epochs50) --out epochs --labels "5 époques" "20 époques" "50 époques"

python compare.py $(last lr1e-4) $B $(last lr5e-4) $(last lr1e-3) $(last lr2e-3) --out lr \
    --labels "lr=1e-4" "lr=2e-4 (baseline)" "lr=5e-4" "lr=1e-3" "lr=2e-3"
python compare.py $(last T100) $(last T200) $(last T500) $B --out T \
    --labels "T=100 (β_end=0.189)" "T=200 (β_end=0.0977)" "T=500 (β_end=0.0398)" "T=1000 (baseline)"
python compare.py $(last beta_end0.01) $B $(last beta_end0.05) --out beta_end \
    --labels "β_end=0.01" "β_end=0.02 (baseline)" "β_end=0.05"
python compare.py $(last beta_start1e-5) $B $(last beta_start1e-3) --out beta_start \
    --labels "β_start=1e-5" "β_start=1e-4 (baseline)" "β_start=1e-3"
python compare.py $(last batch64) $B $(last batch256) --out batch \
    --labels "batch 64" "batch 128 (baseline)" "batch 256"
python compare.py $(last base_ch16) $B $(last base_ch64) --out base_ch \
    --labels "base_ch=16" "base_ch=32 (baseline)" "base_ch=64"
python compare.py $(last emb32) $B $(last emb256) --out emb_dim \
    --labels "emb_dim=32" "emb_dim=128 (baseline)" "emb_dim=256"
python compare.py $B $(last unet_ch40_temoin) $(last resunet) --out architecture \
    --labels "U-Net (baseline, 566k)" "U-Net base_ch=40 (témoin, 855k)" "ResUNet (869k)"
python compare.py $B $(last unet_noskip) $(last unet_not) --out ablations \
    --labels "U-Net complet" "sans skip connections" "sans t"

python loss_par_t.py $B $(last cosine) --out schedule --labels linear cosine
python loss_par_t.py $B $(last unet_not) --out sans_t --labels "avec t" "sans t"
