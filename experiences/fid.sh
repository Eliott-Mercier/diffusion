#!/bin/bash
source "$(dirname "$0")/common.sh"
python fid.py --references \
    $B $(last seed1) $(last seed2) \
    $(last epochs5) $(last epochs50) $(last cosine) \
    $(last unet_noskip) $(last unet_not) \
    $(last resunet) $(last K1_resunet_lr1e-3) \
    $(last baseline28) \
    $(last C1_resunet_lr1e-3_28) $(last C1_resunet_lr1e-3_28_seed1) \
    $(last C2_C1+cosine_28) $(last C2_C1+cosine_28_seed1) \
    $(last C3_C2+batch64_28)
