#!/bin/bash
source "$(dirname "$0")/common.sh"
run name=C1_resunet_lr1e-3_28_seed1 seed=1 "${C[@]}"
run name=C2_C1+cosine_28_seed1 seed=1 "${C[@]}" schedule=cosine
