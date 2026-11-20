#!/bin/bash
source "$(dirname "$0")/common.sh"
python compare.py $B $(last resunet) $(last K1_resunet_lr1e-3) $(last K2_K1+cosine) $(last K3_K2+batch64) \
    $(last K4_K3+emb256) $(last K5_unet64_combo) $(last base_ch64) --out combinations \
    --labels "baseline" "ResUNet" "K1 ResUNet+lr1e-3" "K2 K1+cosine" "K3 K2+batch64" "K4 K3+emb256" \
             "K5 U-Net64+lr+cos+b64+emb256" "U-Net64 only"
python compare.py $B $(last baseline28) --out resolution --labels "baseline 16x16" "baseline 28x28"
python compare.py $(last baseline28) $(last C1_resunet_lr1e-3_28) $(last C2_C1+cosine_28) $(last C3_C2+batch64_28) \
    --out candidates_28 --labels "baseline 28x28 (20 ep.)" "C1 ResUNet+lr1e-3" "C2 C1+cosine" "C3 C2+batch64"
python compare.py $(last C1_resunet_lr1e-3_28) $(last C1_resunet_lr1e-3_28_seed1) $(last C2_C1+cosine_28) \
    $(last C2_C1+cosine_28_seed1) --out C1_vs_C2_seeds --labels "C1 seed 0" "C1 seed 1" "C2 seed 0" "C2 seed 1"
